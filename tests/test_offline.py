"""CPU-only tests: episode bookkeeping, common random numbers, refresh rule, paired statistics,
and the cache-append logic of the stale-vision sampler on a toy attention stack."""

import csv
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evla import common, methods, stats  # noqa: E402


def test_parse_range_and_shards():
    assert common.parse_range("0-3,7,9-10,2") == [0, 1, 2, 3, 7, 9, 10]
    items = [(t, i) for t in range(10) for i in range(50)]
    s0, s1 = common.select_shard(items, "0/2"), common.select_shard(items, "1/2")
    assert len(s0) + len(s1) == 500 and not set(s0) & set(s1)
    assert {t for t, _ in s0} == set(range(10))  # interleaved shards cover every task
    with pytest.raises(ValueError):
        common.parse_range("5-3")


def test_resume_reads_done_pairs(tmp_path):
    p = tmp_path / "r.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "init_id", "success"])
        w.writeheader()
        w.writerow({"task_id": 1, "init_id": 4, "success": 1})
        w.writerow({"task_id": 2, "init_id": 0, "success": 0})
    assert common.load_done(p) == {(1, 4), (2, 0)}
    assert common.load_done(tmp_path / "missing.csv") == set()


def test_crn_noise_depends_only_on_episode_and_step():
    a, b = methods.CRNNoise(), methods.CRNNoise()
    a.set_episode(common.episode_key(3, 7))
    b.set_episode(common.episode_key(3, 7))
    a.step = b.step = 12
    assert torch.equal(a((1, 50, 32), "cpu"), b((1, 50, 32), "cpu"))
    b.step = 13
    assert not torch.equal(a((1, 50, 32), "cpu"), b((1, 50, 32), "cpu"))
    assert len({methods.crn_seed(k, s) for k in range(20) for s in range(600)}) == 20 * 600


def test_fixed_noise_is_constant_within_episode_and_matches_step_zero():
    ref, fx = methods.CRNNoise(), methods.CRNNoise(fixed=True)
    ref.set_episode(common.episode_key(2, 5))
    fx.set_episode(common.episode_key(2, 5))
    n0 = ref((1, 50, 32), "cpu")
    fx.step = 40
    assert torch.equal(fx((1, 50, 32), "cpu"), n0)
    fx.set_episode(common.episode_key(2, 6))
    assert not torch.equal(fx((1, 50, 32), "cpu"), n0)


def test_holm_adjustment():
    from evla import stats

    adj = stats.holm([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_refresh_rule():
    f = methods.should_refresh
    assert f(False, 0, 5, False, None, None)  # no cache yet
    assert f(True, 5, 5, False, None, None)  # period reached
    assert not f(True, 4, 5, False, None, None)
    assert f(True, 1, 5, True, None, None)  # gripper flip forces refresh
    assert f(True, 1, 5, False, 0.2, 0.1)  # image changed beyond tau
    assert not f(True, 1, 5, False, 0.05, 0.1)


def test_gripper_flip_forces_refresh():
    s = methods.StaleVisionSampler.__new__(methods.StaleVisionSampler)
    s.refresh_on_gripper = True
    s.reset()
    s.note_action(torch.tensor([[0.0] * 6 + [-1.0]]))
    assert not s.forced
    s.note_action(torch.tensor([[0.0] * 6 + [-0.8]]))
    assert not s.forced
    s.note_action(torch.tensor([[0.0] * 6 + [1.0]]))
    assert s.forced


def test_mcnemar_and_newcombe():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert math.isclose(stats.mcnemar_exact(0, 6), 2 * 0.5**6)
    d, lo, hi = stats.newcombe_paired(40, 5, 5, 50)
    assert d == 0 and lo < 0 < hi and math.isclose(-lo, hi, rel_tol=1e-9)
    d, lo, hi = stats.newcombe_paired(30, 15, 2, 53)
    assert lo < d < hi and lo > 0  # clear improvement of method A
    for a, b, c, dd in [(0, 0, 0, 10), (10, 0, 0, 0), (3, 7, 0, 0)]:
        _, lo, hi = stats.newcombe_paired(a, b, c, dd)
        assert -1 <= lo <= hi <= 1


def test_compare_direction_and_noninferiority():
    rows_a = [{"task_id": t, "init_id": i, "success": 1} for t in range(4) for i in range(25)]
    rows_b = [dict(r) for r in rows_a]
    rows_b[0]["success"] = 0
    res = stats.compare(rows_a, rows_b, margin=0.05)
    assert res["n_pairs"] == 100 and res["a_only"] == 1 and res["b_only"] == 0
    assert res["diff_b_minus_a"] == pytest.approx(-0.01)
    assert res["ci95_newcombe"][0] < -0.01 < res["ci95_newcombe"][1]


class _Layer(torch.nn.Module):
    """Minimal pre-norm attention block exposing the attributes used by append_prefix_tokens."""

    def __init__(self, d):
        super().__init__()

        class _Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head_dim = d
                self.q_proj = torch.nn.Linear(d, d, bias=False)
                self.k_proj = torch.nn.Linear(d, d, bias=False)
                self.v_proj = torch.nn.Linear(d, d, bias=False)
                self.o_proj = torch.nn.Linear(d, d, bias=False)

        self.self_attn = _Attn()
        self.input_layernorm = torch.nn.LayerNorm(d)
        self.post_attention_layernorm = torch.nn.LayerNorm(d)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, 2 * d), torch.nn.GELU(), torch.nn.Linear(2 * d, d))


class _Cache:
    def __init__(self):
        self.k, self.v = {}, {}

    def update(self, k, v, li):
        if li in self.k:
            self.k[li] = torch.cat([self.k[li], k], dim=2)
            self.v[li] = torch.cat([self.v[li], v], dim=2)
        else:
            self.k[li], self.v[li] = k, v
        return self.k[li], self.v[li]


class _ToyVWE:
    """Toy stand-in for SmolVLMWithExpertModel: self-attention prefill with a K/V cache."""

    def __init__(self, d=8, n_layers=3):
        torch.manual_seed(0)
        self.layers = [_Layer(d) for _ in range(n_layers)]
        self.num_vlm_layers = n_layers
        self.lm_expert = None
        self.d = d

        class _Cfg:
            class text_config:  # noqa: N801
                head_dim = d

        class _Vlm:
            config = _Cfg

        self.vlm = _Vlm

    def get_vlm_model(self):
        class _M:
            text_model = None

        return _M

    def get_model_layers(self, models):
        return [self.layers, [None] * self.num_vlm_layers]

    def forward_attn_layer(self, model_layers, inputs, li, pos, mask, bsz, head_dim, use_cache, past_key_values):
        layer = model_layers[0][li]
        h = layer.input_layernorm(inputs[0])
        q = layer.self_attn.q_proj(h)[:, :, None]  # B,L,H=1,D
        k = layer.self_attn.k_proj(h)[:, :, None]
        v = layer.self_attn.v_proj(h)[:, :, None]
        k, v = past_key_values.update(k.transpose(1, 2), v.transpose(1, 2), li)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        att = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(head_dim)
        att = att.masked_fill(~mask[:, None], float("-inf")).softmax(-1)
        out = torch.einsum("bhqk,bkhd->bqhd", att, v).reshape(q.shape[0], q.shape[1], -1)
        return [out], past_key_values

    def full_prefill(self, embs, mask):
        cache = _Cache()
        h = embs
        for li, layer in enumerate(self.layers):
            (att,), cache = self.forward_attn_layer([self.layers], [h, None], li, None, mask, 1, self.d, True, cache)
            out = layer.self_attn.o_proj(att) + h
            h = layer.mlp(layer.post_attention_layernorm(out)) + out
        return cache


def test_append_prefix_tokens_matches_full_prefill():
    """Image/language tokens never attend to the state token, so prefilling them first and
    appending the state token later must give the same K/V cache as one joint prefill."""
    vwe = _ToyVWE()
    L = 6
    embs = torch.randn(1, L + 1, vwe.d)
    full_mask = torch.ones(1, L + 1, L + 1, dtype=torch.bool)
    full_mask[:, :L, L] = False  # prefix tokens cannot see the state token
    with torch.no_grad():
        ref = vwe.full_prefill(embs, full_mask)
        cache = vwe.full_prefill(embs[:, :L], full_mask[:, :L, :L])
        cache = methods.append_prefix_tokens(vwe, embs[:, L:], full_mask[:, L:, :], None, cache)
    for li in range(vwe.num_vlm_layers):
        assert torch.allclose(ref.k[li], cache.k[li], atol=1e-5)
        assert torch.allclose(ref.v[li], cache.v[li], atol=1e-5)


def test_shift_chunk():
    c = torch.arange(10, dtype=torch.float32).view(1, 5, 2)
    s = methods.shift_chunk(c, 1)
    assert s.shape == c.shape
    assert torch.equal(s[:, :4], c[:, 1:]) and torch.equal(s[:, 4], c[:, 4])
    assert torch.equal(methods.shift_chunk(c, 0), c)


def test_warm_integrate_matches_euler_from_t1_and_is_exact_on_linear_path():
    torch.manual_seed(0)
    noise = torch.randn(1, 50, 32)

    def f(x, t):  # arbitrary smooth field
        return 0.1 * x + t[:, None, None]

    ref, dt = noise.clone(), -1.0 / 4
    for i in range(4):
        ref = ref + dt * f(ref, torch.full((1,), 1.0 + i * dt))
    out = methods.warm_integrate(f, noise, prior=torch.zeros_like(noise), t0=1.0, steps=4)
    assert torch.allclose(out, ref, atol=1e-6)

    # With the exact conditional velocity of the training path, x_t = t*eps + (1-t)*a,
    # u = (x_t - a)/t, a warm start from the true chunk lands exactly on it for any t0 and M.
    a = torch.randn(1, 50, 32)
    for t0, m in [(0.3, 1), (0.5, 3), (0.8, 5)]:
        out = methods.warm_integrate(lambda x, t: (x - a) / t[:, None, None], noise, a, t0, m)
        assert torch.allclose(out, a, atol=1e-5)


class _FakePolicy:
    """Minimal stand-in for SmolVLAPolicy: an action queue refilled with a fixed chunk when empty."""

    def __init__(self, chunk, n_action_steps):
        from collections import deque

        self.chunk = chunk
        self._queues = {"action": deque(maxlen=n_action_steps)}
        self.n_calls = 0

    def select_action(self, batch, noise=None, **kw):
        q = self._queues["action"]
        if not q:
            self.n_calls += 1
            q.extend(self.chunk[: q.maxlen])
        return q.popleft()

    def reset(self):
        self._queues["action"].clear()


def _grip_chunk(signs):
    return [torch.tensor([[0.0] * 6 + [float(s)]]) for s in signs]


def test_event_trigger_replans_before_gripper_switch():
    # chunk: open for 3 steps, then close; executed 10 actions per call
    pol = _FakePolicy(_grip_chunk([-1, -1, -1, 1, 1, 1, 1, 1, 1, 1]), n_action_steps=10)
    trig = methods.install_event_trigger(pol, unnormalize=lambda a: a, hold=5)
    executed = [pol.select_action(None)[0, -1].item() for _ in range(3)]
    assert executed == [-1, -1, -1]
    assert trig.n_events == 1 and len(pol._queues["action"]) == 0  # chunk dropped before the switch
    pol.select_action(None)  # re-plans from the "new observation"
    assert pol.n_calls == 2


def test_event_trigger_hold_and_reset():
    pol = _FakePolicy(_grip_chunk([-1, 1] * 5), n_action_steps=10)  # chattering chunk
    trig = methods.install_event_trigger(pol, unnormalize=lambda a: a, hold=3)
    for _ in range(12):
        pol.select_action(None)
    # with hold=3 a trigger can fire at most once every 4 steps
    assert 1 <= trig.n_events <= 3
    pol.reset()
    assert trig.n_events == 0 and trig.since > 10**8


def test_no_event_without_switch():
    pol = _FakePolicy(_grip_chunk([1] * 10), n_action_steps=10)
    trig = methods.install_event_trigger(pol, unnormalize=lambda a: a, hold=5)
    for _ in range(25):
        pol.select_action(None)
    assert trig.n_events == 0 and pol.n_calls == 3


def test_static_prefix_cache_matches_dynamic_append_and_crop():
    torch.manual_seed(0)

    class _Dyn:
        def __init__(self, layers):
            self.layers = [methods._KV(k, v) for k, v in layers]

    dyn = _Dyn([(torch.randn(1, 2, 7, 4), torch.randn(1, 2, 7, 4)) for _ in range(3)])
    st = methods.StaticPrefixCache(dyn)
    st.load(dyn)
    k_new, v_new = torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)
    k, v = st.update(k_new, v_new, 1)
    assert torch.equal(k, torch.cat([dyn.layers[1].keys, k_new], 2))
    assert torch.equal(v, torch.cat([dyn.layers[1].values, v_new], 2))
    st.crop(7)  # no state change
    assert st.layers[1].keys.shape[2] == 7 and torch.equal(st.layers[1].keys, dyn.layers[1].keys)
    # buffers keep their address when a new prefix is loaded
    ptr = st.layers[0].keys.data_ptr()
    dyn2 = _Dyn([(torch.randn(1, 2, 7, 4), torch.randn(1, 2, 7, 4)) for _ in range(3)])
    st.load(dyn2)
    assert st.layers[0].keys.data_ptr() == ptr and torch.equal(st.layers[0].keys, dyn2.layers[0].keys)


def test_noise_spread_stats():
    import numpy as np

    from evla.noise_spread import first_action_stats

    acts = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0]])
    out = first_action_stats(acts, ref_mean=np.array([1.0, 0.0, 0.5]))
    assert abs(out["grip_split"] - 0.25) < 1e-9
    assert out["spread"] > 0 and abs(out["to_mean10"]) < 1e-9
    same = first_action_stats(np.ones((5, 3)), None)
    assert same["spread"] == 0 and same["grip_split"] == 0 and "to_mean10" not in same
