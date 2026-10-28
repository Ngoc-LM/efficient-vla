"""Build SmolVLA + LIBERO exactly the way `lerobot-eval` (LeRobot v0.6.1) does.

Only the pieces needed for single-episode control are rebuilt here, so that each episode can be
pinned to one LIBERO init state (needed for resumable, sharded, paired evaluation).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch  # noqa: E402

DEFAULT_POLICY = "HuggingFaceVLA/smolvla_libero"


@dataclass
class Bundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    env_preprocessor: Any
    env_postprocessor: Any
    env_cfg: Any
    policy_cfg: Any


def build(
    suite: str,
    policy_path: str = DEFAULT_POLICY,
    device: str = "cuda",
    n_action_steps: int | None = None,
    num_steps: int | None = None,
    dtype: str = "native",
) -> Bundle:
    """Load policy, processors and the LIBERO env config.

    dtype (precision of the SmolVLM backbone + action expert; the small state/action projections
    always stay in float32, as in the released code):
      native   what LeRobot loads: backbone and expert in bfloat16 (emulated on Turing GPUs such as T4)
      fp32     whole policy in float32
      fp16     backbone and expert in float16 (native tensor-core type on T4)
      visfp16  only the vision encoder + connector in float16, the rest as native
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path
    cfg.device = device
    if n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    if num_steps is not None:
        cfg.num_steps = num_steps

    env_cfg = LiberoEnvConfig(task=suite)
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map={})
    policy.eval()
    set_precision(policy, dtype)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    return Bundle(policy, preprocessor, postprocessor, env_pre, env_post, env_cfg, cfg)


def set_precision(policy: Any, dtype: str) -> None:
    vwe = policy.model.vlm_with_expert
    if dtype == "native":
        return
    if dtype == "fp32":
        policy.float()
    elif dtype == "fp16":
        vwe.half()  # SmolVLM (vision + text) and the action expert; projections stay float32
    elif dtype == "visfp16":
        vlm = vwe.get_vlm_model()
        vlm.vision_model.half()
        vlm.connector.half()
        text_dtype = vlm.text_model.get_input_embeddings().weight.dtype
        orig = vwe.embed_image

        def embed_image(image):
            return orig(image).to(text_dtype)

        vwe.embed_image = embed_image
    else:
        raise ValueError(f"unknown dtype mode {dtype!r}")


def make_task_env(env_cfg: Any, task_id: int):
    """One synchronous single-env vector env for (suite, task_id); init state is set per episode."""
    import gymnasium as gym
    from lerobot.envs.libero import create_libero_envs

    kwargs = dict(env_cfg.gym_kwargs)
    kwargs["task_ids"] = [task_id]
    envs = create_libero_envs(
        task=env_cfg.task,
        n_envs=1,
        camera_name=env_cfg.camera_name,
        init_states=env_cfg.init_states,
        gym_kwargs=kwargs,
        env_cls=gym.vector.SyncVectorEnv,
        control_mode=env_cfg.control_mode,
        episode_length=env_cfg.episode_length,
        camera_name_mapping=env_cfg.camera_name_mapping,
    )
    return envs[env_cfg.task][task_id]


def set_init_state(vec_env: Any, init_id: int) -> None:
    """LiberoEnv.reset() uses `init_state_id` and then adds the vector stride (1 here)."""
    inner = vec_env.envs[0].unwrapped
    inner.init_state_id = int(init_id)


def param_dtypes(policy: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in policy.parameters():
        key = str(p.dtype).replace("torch.", "")
        out[key] = out.get(key, 0) + p.numel()
    return out


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
