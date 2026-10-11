#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
task_lab_python="${LAB_PYTHON:-/home/rx/miniconda3/envs/lab_v5.1/bin/python}"
"$task_lab_python" -m venv --system-site-packages .mapping_env
# Keep the PyTorch used by Isaac Sim and Isaac Lab unchanged.
.mapping_env/bin/python -m pip install --no-deps torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.mapping_env/bin/python -m pip install --no-deps --target .deps 'https://github.com/nvidia-isaac/nvblox/releases/download/v0.0.10/nvblox_torch-0.0.10+cu12ubuntu22-py3-none-linux_x86_64.whl'
.mapping_env/bin/python -m pip install --no-deps --target .cuda_extra nvidia-cusparselt-cu12==0.7.1 nvidia-nvshmem-cu12==3.3.20 nvidia-nccl-cu12==2.27.5
task_site="$(.mapping_env/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
task_nvidia_parent="$("$task_lab_python" -c 'import pathlib,nvidia; print(pathlib.Path(nvidia.__file__).parent)')"
if [ ! -e "$task_site/nvidia" ]; then ln -s "$task_nvidia_parent" "$task_site/nvidia"; fi
echo 'Ready. Use ./run.sh all --frames 60'
