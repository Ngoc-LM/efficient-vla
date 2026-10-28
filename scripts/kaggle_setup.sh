#!/usr/bin/env bash
# Environment setup used for every run in this repo (Kaggle, 2x Tesla T4, Python 3.13, CUDA 12.8).
# Versions are pinned on purpose; see README "Environment" for the reasons.
set -euo pipefail

# 1) egl_probe / hf-egl-probe (pulled in by robomimic / hf-libero) build with CMake.
#    pip's isolated build env hides the pip-installed cmake shim, so build them without isolation.
pip install -q "cmake<4"
pip install -q --no-build-isolation egl_probe hf-egl-probe

# 2) LeRobot v0.6.1 (commit 7e241bd) with SmolVLA + LIBERO extras.
#    MuJoCo is pinned below 3.4.0: from 3.4.0 the stored init states of libero_spatial task 5
#    settle differently (lerobot issue #4390), which changes the benchmark itself.
pip install -q "lerobot[smolvla,libero] @ git+https://github.com/huggingface/lerobot.git@v0.6.1" \
  "mujoco>=3.3,<3.4"

# 3) Headless GPU rendering. The Kaggle image only registers the Mesa EGL vendor, so MuJoCo
#    would fall back to software rendering; register the NVIDIA EGL library explicitly.
EGL_LIB=/usr/local/nvidia/lib64/libEGL_nvidia.so.0
if [ -f "$EGL_LIB" ]; then
  echo "{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"$EGL_LIB\"}}" \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi

# 4) LIBERO asks an interactive question on first import; answer "N" to write the default config.
echo N | python -c "import libero.libero" > /dev/null

python - <<'EOF'
import importlib.metadata as md
for p in ["lerobot", "mujoco", "torch", "transformers", "robosuite", "hf-libero", "numpy"]:
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p}: not installed")
EOF
