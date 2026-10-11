#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
task_lab_python="${LAB_PYTHON:-/home/rx/miniconda3/envs/lab_v5.1/bin/python}"
export OMNI_KIT_ACCEPT_EULA=YES
task_cuda_libs="$PWD/.cuda_extra/nvidia/cusparselt/lib:$PWD/.cuda_extra/nvidia/nvshmem/lib:$PWD/.cuda_extra/nvidia/nccl/lib"
case "${1:-help}" in
  capture)
    shift
    exec "$task_lab_python" scripts/isaac_scene.py --headless "$@"
    ;;
  map)
    shift
    export LD_LIBRARY_PATH="$task_cuda_libs:${LD_LIBRARY_PATH:-}"
    # Torch's global dependency preload can find the inherited older NCCL
    # before LD_LIBRARY_PATH. Pin this process to the local compatible copy.
    export LD_PRELOAD="$PWD/.cuda_extra/nvidia/nccl/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
    exec .mapping_env/bin/python scripts/reconstruct.py "$@"
    ;;
  plot)
    shift
    exec "$task_lab_python" scripts/plot_results.py "$@"
    ;;
  validate)
    shift
    exec "$task_lab_python" scripts/validate_results.py "$@"
    ;;
  positions)
    shift
    exec "$task_lab_python" scripts/sample_positions.py "$@"
    ;;
  sphere)
    shift
    exec "$task_lab_python" scripts/isaac_scene.py --view fused --output outputs/positions/pole --sphere-radius 1.2 --export-dir outputs/spherical/view "$@"
    ;;
  live)
    shift
    exec "$task_lab_python" scripts/isaac_scene.py --live --sphere-radius 1.2 --output outputs/live "$@"
    ;;
  view)
    shift
    task_view_mode="${1:-fused}"
    if [ "$#" -gt 0 ]; then shift; fi
    exec "$task_lab_python" scripts/isaac_scene.py --view "$task_view_mode" "$@"
    ;;
  all)
    shift
    task_result_dir="$PWD/outputs/baseline"
    task_all_args=("$@")
    for ((task_i=0; task_i<${#task_all_args[@]}; task_i++)); do
      if [ "${task_all_args[$task_i]}" = --output ]; then
        task_result_dir="${task_all_args[$((task_i+1))]}"
      elif [[ "${task_all_args[$task_i]}" == --output=* ]]; then
        task_result_dir="${task_all_args[$task_i]#--output=}"
      fi
    done
    "$0" capture "$@"
    "$0" map --output "$task_result_dir"
    "$0" validate --output "$task_result_dir"
    "$0" plot --output "$task_result_dir"
    ;;
  *)
    echo 'Usage: ./run.sh {capture|map|plot|validate|positions|sphere|live|view [camera|lidar|fused]|all} [options]'
    ;;
esac
