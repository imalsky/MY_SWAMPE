#!/bin/bash
#SBATCH -J SWAMP_W43B
#SBATCH -o SWAMP_W43B.o%j
#SBATCH -e SWAMP_W43B.e%j
#SBATCH -p gpu
#SBATCH --mem=24G
#SBATCH -t 36:00:00
#SBATCH --gpus=1
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

# =============================================================================
# WASP-43 b real-data pilot retrieval on the JPL edge GPU cluster (SLURM).
# Fetches/prepares the public JWST/MIRI phase-curve data, then delegates to the
# shared retrieval/scripts/run.sh JAX/BlackJAX launcher with the WASP-43b
# config overrides. Mirrors retrieval/wasp_43b_test/run_nas_wasp43b.pbs (the
# NAS/PBS launcher) but adapted for this system's SLURM scheduler + writable
# conda env: no NAS proxy, no --user/PYTHONUSERBASE installs, plain `conda
# activate` + `pip install` straight into the env (same as run.sh).
#
#   cd MY_SWAMP && sbatch retrieval/wasp_43b_test/run_slurm_wasp43b.sh
#
# LIVE PROGRESS: SLURM's own -o/-e files update live, but this also streams
# everything to a dedicated log you can tail the same way as the PBS run:
#   tail -f retrieval/wasp_43b_test/logs/WASP43B_<jobid>.log
#
# Walltime is 36h (vs run.sh's generic 6h) to match run_nas_wasp43b.pbs -- the
# WASP-43b config (dt=120s, 20-day spin-up, 256 particles) is heavier than the
# default synthetic-data preset.
#
# Env (-v):
#   CONDA_ENV                 conda env for fetch/prepare + the delegated
#                              run.sh (default: swamp, same as run.sh)
#   SWAMP_RETRIEVAL_PRESET     fast | gpu | prod   (default: gpu)
#   SWAMP_RETRIEVAL_USE_X64    0 | 1               (default: 1)
#   SWAMP_SKIP_INSTALL / SWAMP_SKIP_PLOTS   passed through to run.sh unchanged
# =============================================================================
set -euo pipefail

# --- locate the repo (this script lives in retrieval/wasp_43b_test/) --------
if [ -n "${PROJECT_ROOT:-}" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/retrieval/wasp_43b_test" ]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"                                    # submitted from MY_SWAMP/
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] \
     && [ "$(basename "${SLURM_SUBMIT_DIR}")" = "wasp_43b_test" ]; then
  PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR}/../.." && pwd -P)"         # submitted from retrieval/wasp_43b_test/
else
  _sd="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  PROJECT_ROOT="$(cd -- "${_sd}/../.." && pwd -P)"                      # direct bash, not sbatch
fi
SUITE_DIR="${PROJECT_ROOT}/retrieval/wasp_43b_test"
RETRIEVAL_DIR="${PROJECT_ROOT}/retrieval/scripts"
if [ ! -d "${SUITE_DIR}" ] || [ ! -d "${RETRIEVAL_DIR}" ]; then
  echo "ERROR: could not locate retrieval/wasp_43b_test and retrieval/scripts under PROJECT_ROOT=${PROJECT_ROOT}"
  exit 1
fi

# --- stream all output to a tail-able log ------------------------------------
LOG_DIR="${SUITE_DIR}/logs"
mkdir -p "${LOG_DIR}"
_jobid="${SLURM_JOB_ID:-manual_$$}"
LOG_FILE="${LOG_DIR}/WASP43B_${_jobid}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Live log: tail -f ${LOG_FILE}"
echo "======================================================"
echo "  Job info:  host=$(hostname)  SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}  $(date)"
echo "  PROJECT_ROOT=${PROJECT_ROOT}"
echo "======================================================"

# --- conda env (fetch/prepare need numpy + h5py; run.sh activates it again) --
CONDA_ENV="${CONDA_ENV:-swamp}"
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  echo "conda env: ${CONDA_ENV}  ($(which python))"
fi
python -c "import h5py" 2>/dev/null || python -m pip install -U h5py

# --- fetch + prepare the WASP-43 b JWST/MIRI data ----------------------------
cd -P "${SUITE_DIR}"
python -u scripts/fetch_wasp43b_data.py
python -u scripts/prepare_wasp43b_observations.py

# --- delegate to the shared JAX/BlackJAX SLURM launcher with WASP-43b config -
export PROJECT_ROOT
export SWAMP_RETRIEVAL_PRESET="${SWAMP_RETRIEVAL_PRESET:-gpu}"
export SWAMP_RETRIEVAL_USE_X64="${SWAMP_RETRIEVAL_USE_X64:-1}"
export SWAMP_RETRIEVAL_OVERRIDES_FILE="${SUITE_DIR}/config/wasp43b_pilot_gpu.json"
export SWAMP_PLOT_OUT_DIR="${SUITE_DIR}/outputs"
export SWAMP_PLOTS_DIR="${SUITE_DIR}/plots"

bash "${RETRIEVAL_DIR}/run.sh"
