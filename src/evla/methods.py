"""Inference-time variants of SmolVLA used in the study, plus instrumentation.

All variants keep the released weights unchanged. They are installed by patching methods on a
loaded `SmolVLAPolicy`, so the LeRobot rollout loop (`lerobot_eval.rollout`) is reused as is.

Variants
--------
baseline        released config (n_action_steps=1 for HuggingFaceVLA/smolvla_libero, 10 flow steps).
open_loop K     execute K actions of each chunk before re-planning (`n_action_steps=K`).
flow S          S Euler steps instead of 10.
stale K         *stale-vision, fresh-state*: re-plan every control step, but recompute the
                vision encoder + image/language prefix only every K steps (or on a trigger).
                In between, only the proprioceptive state token is pushed through the VLM
                (attending to the cached image/language keys/values) and the action expert runs.
                This is exact when K=1, because image/language tokens never attend to the state
                token in SmolVLA's prefix mask.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812

# ----------------------------------------------------------------------------------------------
# Common random numbers: identical flow-matching noise for every method at a given control step.
# ----------------------------------------------------------------------------------------------


class CRNNoise:
    """Replaces `model.sample_noise`; noise depends only on (episode_key, control_step)."""

    def __init__(self) -> None:
        self.episode_key = 0
        self.step = 0

    def set_episode(self, key: int) -> None:
        self.episode_key = int(key)
        self.step = 0

    def __call__(self, shape, device):
        g = torch.Generator(device="cpu")
        g.manual_seed(crn_seed(self.episode_key, self.step))
        return torch.randn(shape, generator=g, dtype=torch.float32).to(device)


def crn_seed(episode_key: int, step: int) -> int:
    # Deterministic, collision-free for step < 10_000.
    return (int(episode_key) * 10_000 + int(step)) % (2**63 - 1)


# ----------------------------------------------------------------------------------------------
# Instrumentation
# ----------------------------------------------------------------------------------------------


@dataclass
class CallStats:
    n_chunk_calls: int = 0  # policy forward passes that produced an action chunk
    n_vision_calls: int = 0  # vision encoder + full prefix recomputations
    policy_s: float = 0.0  # synchronized wall time inside chunk computation
    per_call_ms: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.n_chunk_calls = 0
        self.n_vision_calls = 0
        self.policy_s = 0.0
        self.per_call_ms = []


def instrument(policy, stats: CallStats, crn: CRNNoise | None = None, sampler=None) -> None:
    """Wrap `_get_action_chunk` (called only when a new chunk is computed) and `select_action`."""
    orig_chunk = policy._get_action_chunk
    orig_select = policy.select_action

    def timed_chunk(batch, noise=None, **kw):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_chunk(batch, noise, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        stats.n_chunk_calls += 1
        stats.policy_s += dt
        stats.per_call_ms.append(dt * 1e3)
        if sampler is None:
            stats.n_vision_calls += 1
        return out

    def counted_select(batch, noise=None, **kw):
        out = orig_select(batch, noise, **kw)
        if crn is not None:
            crn.step += 1
        return out

    policy._get_action_chunk = timed_chunk
    policy.select_action = counted_select
    if crn is not None:
        policy.model.sample_noise = crn


# ----------------------------------------------------------------------------------------------
# Stale-vision, fresh-state sampler
# ----------------------------------------------------------------------------------------------


def thumbnails(images: list[torch.Tensor], size: int = 16) -> torch.Tensor:
    """Cheap scene signature: per-camera average-pooled thumbnails, concatenated on channels."""
    return torch.cat([F.adaptive_avg_pool2d(im.float(), size) for im in images], dim=1)


def should_refresh(
    has_cache: bool,
    steps_since: int,
    refresh_every: int,
    forced: bool,
    img_change: float | None,
    img_tau: float | None,
) -> bool:
    """Refresh rule (pure function, unit-tested on CPU)."""
    if not has_cache or forced or steps_since >= refresh_every:
        return True
    return img_tau is not None and img_change is not None and img_change > img_tau


def warm_integrate(denoise_fn, noise, prior, t0: float, steps: int):
    """Euler integration of the flow from t0 (not 1) to 0, starting from a noised prior chunk.

    SmolVLA trains on x_t = t * eps + (1 - t) * a with velocity target u = eps - a and samples by
    integrating from t = 1 (pure noise) to t = 0. Warm start places the previous chunk, shifted by
    the executed actions, at time t0 with the same interpolation, then integrates the remaining
    [t0, 0] interval with `steps` Euler steps (SDEdit-style initialisation for action chunks).
    """
    bsize, device = noise.shape[0], noise.device
    x = t0 * noise + (1.0 - t0) * prior
    dt = -t0 / steps
    for i in range(steps):
        t = torch.full((bsize,), t0 + i * dt, dtype=torch.float32, device=device)
        x = x + dt * denoise_fn(x, t)
    return x


def shift_chunk(chunk: torch.Tensor, n: int) -> torch.Tensor:
    """Drop the n executed actions and repeat the last one to keep the chunk length."""
    if n <= 0:
        return chunk
    n = min(n, chunk.shape[1] - 1)
    return torch.cat([chunk[:, n:], chunk[:, -1:].expand(-1, n, -1)], dim=1)


class FastSampler:
    """Replacement for `VLAFlowMatching.sample_actions` with two independent, combinable options.

    refresh_every=K  reuse the image/language KV cache for K-1 control steps (vision is recomputed
                     every K steps, on a gripper flip, or when the image changes by more than tau);
                     K=1 runs the original prefix path unchanged.
    warm_t0, warm_steps  warm-start the flow from the previous chunk at time t0 with `warm_steps`
                     Euler steps; the first call of an episode and the call after a gripper flip use
                     the full cold integration (config.num_steps from t=1).
    """

    def __init__(
        self,
        model,
        stats: CallStats,
        refresh_every: int = 1,
        img_tau: float | None = None,
        refresh_on_gripper: bool = True,
        warm_t0: float | None = None,
        warm_steps: int | None = None,
        shift: int = 1,
    ) -> None:
        self.model = model
        self.stats = stats
        self.refresh_every = int(refresh_every)
        self.img_tau = img_tau
        self.refresh_on_gripper = refresh_on_gripper
        self.warm_t0 = warm_t0
        self.warm_steps = warm_steps
        self.shift = int(shift)
        self.reset()

    def reset(self) -> None:
        self.cache = None
        self.il_pad = None
        self.L_il = 0
        self.steps_since = 0
        self.ref_thumb = None
        self.forced = False  # vision refresh forced (gripper flip)
        self.forced_cold = False  # cold integration forced (gripper flip)
        self.last_grip = None
        self.prev = None
        self.n_refresh_trigger = {"init": 0, "period": 0, "image": 0, "gripper": 0}
        self.n_warm = 0
        self.n_cold = 0

    def note_action(self, action: torch.Tensor) -> None:
        """Called with every executed env action (after unnormalisation, gripper in [-1, 1]);
        a gripper open/close flip forces a refresh at the next control step."""
        if not self.refresh_on_gripper:
            return
        g = torch.sign(action[..., -1]).detach()
        if self.last_grip is not None and bool((g != self.last_grip).any()):
            self.forced = True
            self.forced_cold = True
        self.last_grip = g

    # -- prefix -------------------------------------------------------------------------------
    def _prefix_full(self, images, img_masks, lang_tokens, lang_masks, state):
        """Original SmolVLA prefix: one joint prefill of image, language and state tokens."""
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        m = self.model
        embs, pad, att = m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        att2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        _, cache = m.vlm_with_expert.forward(
            attention_mask=att2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=m.config.use_cache,
        )
        self.stats.n_vision_calls += 1
        self.n_refresh_trigger["period"] += 1
        return pad, cache

    def _prefix_stale(self, images, img_masks, lang_tokens, lang_masks, state):
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        m = self.model
        vwe = m.vlm_with_expert
        bsize, device = state.shape[0], state.device
        img_change = None
        if self.cache is not None and self.img_tau is not None:
            img_change = float((thumbnails(images) - self.ref_thumb).abs().mean())
        refresh = should_refresh(
            self.cache is not None, self.steps_since, self.refresh_every, self.forced, img_change, self.img_tau
        )
        if refresh:
            reason = (
                "init" if self.cache is None
                else "gripper" if self.forced
                else "period" if self.steps_since >= self.refresh_every
                else "image"
            )
            self.n_refresh_trigger[reason] += 1
            embs, pad, att = m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
            if m.config.prefix_length > 0:
                raise NotImplementedError("prefix padding is not supported by the stale sampler")
            L = embs.shape[1] - 1  # last prefix token is the state token
            il_embs, il_pad, il_att = embs[:, :L], pad[:, :L], att[:, :L]
            att2d = make_att_2d_masks(il_pad, il_att)
            pos = torch.cumsum(il_pad, dim=1) - 1
            _, cache = vwe.forward(
                attention_mask=att2d, position_ids=pos, past_key_values=None,
                inputs_embeds=[il_embs, None], use_cache=True,
            )
            self.cache, self.il_pad, self.L_il = cache, il_pad, L
            self.steps_since = 0
            self.forced = False
            self.ref_thumb = thumbnails(images) if self.img_tau is not None else None
            self.stats.n_vision_calls += 1
            state_emb = embs[:, L:]  # same tensor (and dtype) the full prefix would use
        else:
            self.cache.crop(self.L_il)
            state_emb = m.state_proj(state)
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
            # match the dtype the concatenated prefix would have had
            state_emb = state_emb.to(torch.promote_types(state_emb.dtype, self._il_dtype()))

        n_valid = self.il_pad.sum(dim=1, keepdim=True).long()  # (B, 1): position id of the state token
        mask = torch.cat(
            [self.il_pad[:, None, :], torch.ones(bsize, 1, 1, dtype=torch.bool, device=device)], dim=2
        )
        cache = append_prefix_tokens(vwe, state_emb, mask, n_valid, self.cache)
        prefix_pad = torch.cat([self.il_pad, torch.ones(bsize, 1, dtype=torch.bool, device=device)], dim=1)
        self.steps_since += 1
        return prefix_pad, cache

    # -- full call ------------------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        from lerobot.policies.common.flow_matching import euler_integrate

        m = self.model
        bsize, device = state.shape[0], state.device
        if noise is None:
            noise = m.sample_noise((bsize, m.config.chunk_size, m.config.max_action_dim), device)
        if self.refresh_every <= 1:
            prefix_pad, cache = self._prefix_full(images, img_masks, lang_tokens, lang_masks, state)
            self.forced = False
        else:
            prefix_pad, cache = self._prefix_stale(images, img_masks, lang_tokens, lang_masks, state)

        def denoise(x_t, t):
            return m.denoise_step(x_t=x_t, prefix_pad_masks=prefix_pad, past_key_values=cache, timestep=t)

        if self.warm_t0 is not None and self.prev is not None and not self.forced_cold:
            x = warm_integrate(denoise, noise, shift_chunk(self.prev, self.shift), self.warm_t0, self.warm_steps)
            self.n_warm += 1
        else:
            x = euler_integrate(denoise, noise, m.config.num_steps)
            self.n_cold += 1
            self.forced_cold = False
        self.prev = x
        return x

    def _il_dtype(self):
        # dtype of image/language embeddings produced by the VLM embedding layers
        return self.model.vlm_with_expert.get_vlm_model().text_model.get_input_embeddings().weight.dtype


class StaleVisionSampler(FastSampler):
    """Vision reuse only (kept for the profiler and tests)."""

    def __init__(self, model, stats, refresh_every: int = 5, img_tau=None, refresh_on_gripper: bool = True):
        super().__init__(model, stats, refresh_every=max(2, int(refresh_every)), img_tau=img_tau,
                         refresh_on_gripper=refresh_on_gripper)


def append_prefix_tokens(vwe, embs, attention_mask, position_ids, cache):
    """Push extra prefix tokens through every VLM layer, appending their K/V to `cache`.

    Mirrors the prefix branch of `SmolVLMWithExpertModel.forward` (self-attention in every layer
    during prefill) but starts from an already-filled cache.
    """
    models = [vwe.get_vlm_model().text_model, vwe.lm_expert]
    model_layers = vwe.get_model_layers(models)
    head_dim = vwe.vlm.config.text_config.head_dim
    h = embs
    for li in range(vwe.num_vlm_layers):
        att_outputs, cache = vwe.forward_attn_layer(
            model_layers,
            [h, None],
            li,
            position_ids,
            attention_mask,
            h.shape[0],
            head_dim,
            use_cache=True,
            past_key_values=cache,
        )
        layer = model_layers[0][li]
        att = att_outputs[0]
        if att.dtype != layer.self_attn.o_proj.weight.dtype:
            att = att.to(layer.self_attn.o_proj.weight.dtype)
        out = layer.self_attn.o_proj(att[:, : h.shape[1]])
        out += h
        res = out.clone()
        out = layer.post_attention_layernorm(out)
        out = layer.mlp(out)
        out += res
        h = out
    return cache


class EventTrigger:
    """Gripper-event-triggered re-planning for open-loop chunk execution.

    The policy executes up to n_action_steps actions of each chunk open loop. After an action is
    taken from the queue, if the next queued action switches the gripper (open <-> close) relative to
    the action just executed, the rest of the chunk is dropped, so the policy re-plans from the newest
    observation exactly at the contact event. To avoid re-creating gripper chatter, no new trigger is
    allowed for `hold` control steps after a triggered re-plan.

    `unnormalize` maps a normalised action (B, A) to environment units, where the gripper command is
    the last dimension in [-1, 1] (closing > 0 in LIBERO).
    """

    def __init__(self, policy, unnormalize, hold: int = 5, action_key: str = "action") -> None:
        self.policy = policy
        self.unnormalize = unnormalize
        self.hold = int(hold)
        self.action_key = action_key
        self.reset()

    def reset(self) -> None:
        self.since = 10**9
        self.n_events = 0

    def after_select(self, action: torch.Tensor) -> None:
        q = self.policy._queues[self.action_key]
        self.since += 1
        if not q or self.since <= self.hold:
            return
        g_now = torch.sign(self.unnormalize(action)[..., -1])
        g_next = torch.sign(self.unnormalize(q[0])[..., -1])
        if bool((g_now != g_next).any()):
            q.clear()
            self.n_events += 1
            self.since = 0


def install_event_trigger(policy, unnormalize, hold: int = 5) -> EventTrigger:
    trig = EventTrigger(policy, unnormalize, hold=hold)
    orig_select = policy.select_action
    orig_reset = policy.reset

    def select_action(batch, noise=None, **kw):
        a = orig_select(batch, noise, **kw)
        trig.after_select(a)
        return a

    def reset():
        orig_reset()
        trig.reset()

    policy.select_action = select_action
    policy.reset = reset
    return trig


def install_sampler(policy, stats: CallStats, **kw) -> FastSampler:
    sampler = FastSampler(policy.model, stats, shift=policy.config.n_action_steps, **kw)
    policy.model.sample_actions = sampler.sample_actions
    orig_reset = policy.reset

    def reset():
        orig_reset()
        sampler.reset()

    policy.reset = reset
    return sampler
