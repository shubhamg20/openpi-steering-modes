from collections.abc import MutableMapping
import logging
import math
from typing import Any

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.transformers_replace.models.siglip import check


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_dct_basis(horizon: int, k: int, device, dtype=torch.float32) -> torch.Tensor:
    """
    Orthonormal DCT-II basis matrix B of shape (H, K).
    """
    H = int(horizon)
    K = int(k)
    t = torch.arange(H, device=device, dtype=dtype).view(H, 1)  # (H,1)
    kk = torch.arange(K, device=device, dtype=dtype).view(1, K)  # (1,K)

    B = torch.cos((math.pi / H) * (t + 0.5) * kk)  # (H,K)

    # Orthonormal scaling
    B[:, 0] *= math.sqrt(1.0 / H)
    if K > 1:
        B[:, 1:] *= math.sqrt(2.0 / H)
    return B


def expand_blocks(c: torch.Tensor, horizon: int) -> torch.Tensor:
    """Expand K block parameters across the horizon by repetition."""
    k = c.shape[-2]
    block = max(1, horizon // k)
    expanded = torch.repeat_interleave(c, repeats=block, dim=-2)
    current = expanded.shape[-2]
    if current < horizon:
        pad_len = horizon - current
        pad = c[..., -1:, :].expand(*expanded.shape[:-2], pad_len, c.shape[-1])
        expanded = torch.cat([expanded, pad], dim=-2)
    return expanded[..., :horizon, :]


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config, compile_mode: str | None = "max-autotune"):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Clear GPU memory before model creation to help with CUDA graph recording
        torch.cuda.empty_cache()

        paligemma_model = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        if compile_mode == "max-autotune":
            self.paligemma_with_expert = torch.compile(
                paligemma_model,
                mode=compile_mode,
                fullgraph=True,
            )
            logging.info("torch.compile enabled for paligemma_with_expert (mode: %s)", compile_mode)
        else:
            self.paligemma_with_expert = paligemma_model
            logging.info("torch.compile disabled for paligemma_with_expert")

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)
        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")

        if compile_mode is not None:
            if compile_mode == "max-autotune-no-gemm":
                # This is a custom compile mode that fits into weak GPUs such as RTX 6000 ada.
                self.sample_actions = torch.compile(
                    self.sample_actions,
                    backend="inductor",
                    options={
                        # Note: we're still applying default options, e.g., triton.cudagraphs: True
                        "max_autotune": True,  # Optimize pointwise/reduction ops (Safe)
                        "max_autotune_gemm": False,  # Explicitly DISABLE the memory-hungry GEMM tuner
                    },
                )
            else:
                self.sample_actions = torch.compile(self.sample_actions, mode=compile_mode)
            logging.info("torch.compile enabled for sample_actions (mode: %s)", compile_mode)
        else:
            self.sample_actions = torch.compile(self.sample_actions, mode=compile_mode)
        # Gradient checkpointing is intentionally disabled in this fast variant.
        self.gradient_checkpointing_enabled = False

        if not check.check_whether_transformers_replace_is_installed_correctly():
            raise ValueError("transformers_replace is not set up correctly.")

    def _state_dict_has_paligemma_orig_mod(self, state_dict: MutableMapping[str, Any]) -> bool:
        paligemma_orig_prefix = "paligemma_with_expert._orig_mod."
        return any(key.startswith(paligemma_orig_prefix) for key in state_dict)

    def _convert_paligemma_state_dict_keys(
        self, state_dict: MutableMapping[str, Any], *, insert_orig_mod: bool
    ) -> MutableMapping[str, Any]:
        paligemma_prefix = "paligemma_with_expert."
        paligemma_orig_prefix = f"{paligemma_prefix}_orig_mod."
        converted_state_dict = state_dict.__class__()  # Preserve ordered dicts
        for key, value in state_dict.items():
            new_key = key
            if insert_orig_mod:
                if key.startswith(paligemma_prefix) and not key.startswith(paligemma_orig_prefix):
                    suffix = key[len(paligemma_prefix) :]
                    new_key = f"{paligemma_orig_prefix}{suffix}"
            elif key.startswith(paligemma_orig_prefix):
                suffix = key[len(paligemma_orig_prefix) :]
                new_key = f"{paligemma_prefix}{suffix}"
            converted_state_dict[new_key] = value
        return converted_state_dict

    def load_state_dict(
        self, state_dict: MutableMapping[str, Any], *, strict: bool = True, assign: bool = False
    ) -> Any:
        expects_orig_mod = hasattr(self.paligemma_with_expert, "_orig_mod")
        state_has_orig_mod = self._state_dict_has_paligemma_orig_mod(state_dict)
        converted_state_dict = state_dict
        if expects_orig_mod and not state_has_orig_mod:
            converted_state_dict = self._convert_paligemma_state_dict_keys(state_dict, insert_orig_mod=True)
        elif not expects_orig_mod and state_has_orig_mod:
            converted_state_dict = self._convert_paligemma_state_dict_keys(state_dict, insert_orig_mod=False)
        return super().load_state_dict(converted_state_dict, strict=strict, assign=assign)

    def gradient_checkpointing_enable(self):
        """No-op placeholder to maintain interface compatibility."""
        logging.info("PI0PytorchFast ignores gradient checkpointing requests.")
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_disable(self):
        """Ensure gradient checkpointing stays disabled."""
        logging.info("PI0PytorchFast keeps gradient checkpointing disabled.")
        self.gradient_checkpointing_enabled = False

    def is_gradient_checkpointing_enabled(self):
        """Fast variant never enables checkpointing."""
        return False

    def _select_attention_backend(self):
        """Pick the fastest available attention backend (prefers torch SDPA/flash on GPU)."""
        return "eager" if torch.cuda.is_available() else "eager"

    def _set_attention_backend(self, *, use_expert: bool):
        backend = self._select_attention_backend()
        target_config = (
            self.paligemma_with_expert.gemma_expert.model.config
            if use_expert
            else self.paligemma_with_expert.paligemma.language_model.config
        )
        target_config._attn_implementation = backend  # noqa: SLF001

    ##### DATA DEPENDENT FUNCTION ######
    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        # Create mask in the model's dtype (e.g., bfloat16) to match query dtype in attention
        model_dtype = self.state_proj.weight.dtype
        zero = torch.tensor(0.0, dtype=model_dtype, device=att_2d_masks.device)
        neg_inf = torch.tensor(-2.3819763e38, dtype=model_dtype, device=att_2d_masks.device)
        return torch.where(att_2d_masks_4d, zero, neg_inf)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            # Convert state to match the model's dtype (e.g., bfloat16 or float32)
            state = state.to(self.state_proj.weight.dtype)

            state_emb = self.state_proj(state)
            embs.append(state_emb[:, None, :])
            state_batch = state_emb.shape[0]
            device = state_emb.device
            state_mask = torch.ones(state_batch, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)
            action_time_emb = self.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)
            action_time_emb = self.action_time_mlp_out(action_time_emb)
            adarms_cond = None
        else:
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)  # swish == silu
            x = self.time_mlp_out(x)
            time_emb = F.silu(x)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_embs = prefix_embs.clone()
        # Ensure dtype matches model weights (bfloat16 or float32)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        torch.compiler.cudagraph_mark_step_begin()
        paligemma_outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=adarms_cond,
        )
        suffix_out = paligemma_outputs[1].clone()
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        # B, T, A
        loss = F.mse_loss(u_t, v_t, reduction="none")
        if observation.action_is_pad is not None:
            # B, T, 1
            action_is_not_pad = (~observation.action_is_pad).float().unsqueeze(-1)
            loss = loss * action_is_not_pad
            # B, 1, A
            loss = loss.sum(1, keepdim=True) / action_is_not_pad.sum(1, keepdim=True)
        return loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_embs = prefix_embs.clone()
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self._set_attention_backend(use_expert=False)

        torch.compiler.cudagraph_mark_step_begin()
        torch.compiler.cudagraph_mark_step_begin()
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    @torch.no_grad()
    def sample_noise_action(self, device, observation, action, num_steps=10) -> Tensor:
        """Do a full inference backward and compute the noise (batch_size x num_steps x num_motors)"""
        # print("sample noise action called")
        bsize = observation.state.shape[0]

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_embs = prefix_embs.clone()
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self._set_attention_backend(use_expert=False)

        torch.compiler.cudagraph_mark_step_begin()
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = 1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = action
        time = torch.tensor(0.0, dtype=torch.float32, device=device)
        while time < 1 - dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    @torch.no_grad()
    def sample_noise_action_implicit(
        self,
        device,
        observation,
        action,
        num_steps: int = 10,
        num_fp_iters: int = 8,
        tol: float | None = None,
    ) -> torch.Tensor:
        """
        Invert the flow: Action (t=0) -> Noise (t=1) using implicit Euler
        with fixed-point iteration.
        """
        bsize = observation.state.shape[0]

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_embs = prefix_embs.clone()
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self._set_attention_backend(use_expert=False)

        torch.compiler.cudagraph_mark_step_begin()
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # === implicit Euler, your tensor loop style ===
        dt = 1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = action
        time = torch.tensor(0.0, dtype=torch.float32, device=device)

        while time < 1 - dt/2:
            # next time is a tensor too
            t_next = (time + dt).expand(bsize)

            # fixed-point iteration to solve x_new = x_t + dt*v(x_new, t_next)
            x_new = x_t.clone()
            for i in range(num_fp_iters):
                x_prev = x_new
                torch.compiler.cudagraph_mark_step_begin()
                v_t = self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_new,
                    t_next,    # NOTE: implicit uses t_next, not time
                )
                v_t = v_t.clone()
                x_new = x_t + dt * v_t

                if tol is not None:
                    diff = (x_new - x_prev).abs().max()
                    if diff.item() < tol:
                        break
            x_t = x_new
            time = time + dt     # tensor update keeps your style
        return x_t

    @torch.no_grad()
    def cem_fit_noise_blocks(self,
        sample_actions_fn,
        actions_true: torch.Tensor,  # (H, A)
        *,
        c_init: torch.Tensor | None = None,
        true_action_dim: int = 14,
        action_dim: int = 32,
        horizon: int = 50,
        k_blocks: int = 1,
        iters: int = 50,
        pop: int = 128,
        elites: int = 16,
        init_std: float = 0.5,
        min_std: float = 0.05,
        alpha: float = 0.25,
        verbose: bool = False,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, float]:
        """0th-order CEM search over block-parameterized noise."""
        actions_true = actions_true.to(device)

        if c_init is None:
            mean = torch.zeros(k_blocks, action_dim, device=device)
        else:
            mean = c_init.to(device).clone()
        std = torch.ones_like(mean) * init_std

        best_loss = float("inf")
        best_c = mean.clone()

        for i in range(iters):
            eps = torch.randn(pop, k_blocks, action_dim, device=device)
            c = mean[None] + std[None] * eps

            # Keep current mean as a candidate for stability.
            c[0] = mean

            noise = expand_blocks(c, horizon)  # (pop, H, A)
            actions_hat = sample_actions_fn(noise)  # (pop, H, A)

            loss = (
                (actions_hat[..., :true_action_dim] - actions_true[None, :, :true_action_dim]) ** 2
            ).mean(dim=(1, 2))

            elite_idx = torch.topk(-loss, k=elites).indices
            c_elite = c[elite_idx]

            min_i = torch.argmin(loss)
            loss_min_val = float(loss[min_i])
            if loss_min_val < best_loss:
                best_loss = loss_min_val
                best_c = c[min_i].clone()

            new_mean = c_elite.mean(dim=0)
            new_std = c_elite.std(dim=0).clamp_min(min_std)

            mean = (1 - alpha) * mean + alpha * new_mean
            std = (1 - alpha) * std + alpha * new_std

        if verbose:
            print(
                f"[cem] iter={i:02d} best_loss={best_loss:.6f} "
                f"elite_loss={loss_min_val:.6f} std_mean={std.mean().item():.4f}"
            )

        best_noise = expand_blocks(best_c[None], horizon)[0]
        return best_noise, best_loss

    @torch.no_grad()
    def cem_fit_noise_dct_coeff(
        self,
        sample_actions_fn,
        actions_true: torch.Tensor,  # (H, A)
        *,
        c_init: torch.Tensor | None = None,  # (K, A)
        true_action_dim: int = 14,
        action_dim: int = 32,
        horizon: int = 50,
        k_dct: int = 12,
        iters: int = 30,
        pop: int = 128,
        elites: int = 16,
        init_std: float = 0.5,
        min_std: float = 0.05,
        alpha: float = 0.25,
        verbose: bool = False,
        device: str | torch.device = "cuda",
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        """
        CEM over low-dim DCT coefficients C (K x A), where noise trajectory is:
            N(H x A) = B(H x K) @ C(K x A)
        """
        device = torch.device(device) if isinstance(device, str) else device

        if actions_true.ndim != 2:
            raise ValueError(f"actions_true must be (H,A). Got {tuple(actions_true.shape)}")
        actions_true = actions_true.to(device)

        K = int(k_dct)
        H = int(horizon)
        A = int(action_dim)

        # Precompute basis once (float32 is fine; noise is float32 anyway in your sampler)
        B = make_dct_basis(H, K, device=device, dtype=torch.float32)  # (H,K)

        if c_init is None:
            mean = torch.zeros(K, A, device=device)
        else:
            if c_init.shape != (K, A):
                raise ValueError(f"c_init must be (K,A)=({K},{A}), got {tuple(c_init.shape)}")
            mean = c_init.to(device).clone()

        std = torch.ones_like(mean) * float(init_std)

        best_loss = float("inf")
        best_C = mean.clone()

        for it in range(int(iters)):
            eps = torch.randn(pop, K, A, device=device)
            C = mean[None] + std[None] * eps  # (pop,K,A)

            # Keep current mean as candidate for stability
            C[0] = mean

            # Build noise: N = B @ C -> (pop,H,A)
            noise = torch.matmul(B, C)

            # Decode actions with frozen Pi0 (wrapped)
            actions_hat = sample_actions_fn(noise)  # (pop,H,A)

            # Loss only on real action dims
            loss = ((actions_hat[..., :true_action_dim] - actions_true[None, :, :true_action_dim]) ** 2).mean(
                dim=(1, 2)
            )  # (pop,)

            # Track best sample
            min_i = torch.argmin(loss)
            loss_min = float(loss[min_i])
            if loss_min < best_loss:
                best_loss = loss_min
                best_C = C[min_i].clone()

            # Elite update
            elite_idx = torch.topk(-loss, k=int(elites)).indices
            C_elite = C[elite_idx]
            new_mean = C_elite.mean(dim=0)
            new_std = C_elite.std(dim=0).clamp_min(float(min_std))

            mean = (1 - float(alpha)) * mean + float(alpha) * new_mean
            std = (1 - float(alpha)) * std + float(alpha) * new_std

            if verbose:
                print(
                    f"[cem-dct] it={it:02d} best={best_loss:.6f} "
                    f"elite_best={loss_min:.6f} std_mean={std.mean().item():.4f}"
                )

        best_noise = torch.matmul(B, best_C)  # (H,A)
        return best_noise, best_loss, best_C

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self._set_attention_backend(use_expert=True)

        torch.compiler.cudagraph_mark_step_begin()
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1].clone()
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
