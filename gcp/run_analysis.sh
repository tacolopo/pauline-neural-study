#!/usr/bin/env bash
# =============================================================================
# Pauline Neural Study - Run Analysis Script
# =============================================================================
#
# Executes the complete Pauline Neural Study pipeline on a GCP VM.
#
# Usage:
#   bash gcp/run_analysis.sh              # Full run (default config)
#   bash gcp/run_analysis.sh --quick      # Quick test run
#   bash gcp/run_analysis.sh --phases corpus bootstrap embeddings
#   bash gcp/run_analysis.sh --with-gpu   # Include VAE phase
#
# Prerequisites:
#   - Run gcp/setup_vm.sh first
#   - Corpus data should be in data/ directory
#
# Output:
#   All results are saved to output/ directory:
#     - corpus_summary.json         Corpus statistics
#     - embeddings_*.npz            Trained word embeddings per level
#     - cross_epistle_results.json  Cross-epistle stability analysis
#     - generated_sentences.txt     Synthetic Pauline sentences
#     - fractal_results.json        Self-similarity measurements
#     - bayesian_topics.json        Latent theological topics
#     - pipeline_summary.json       Full run summary
#     - *.png                       Visualization plots
#
# =============================================================================

set -euo pipefail

# --- Configuration ---
PROJECT_DIR="${HOME}/pauline-neural-study"
VENV_DIR="${PROJECT_DIR}/.venv"
CONFIG_FILE="${PROJECT_DIR}/configs/default.yaml"
LOG_FILE="${PROJECT_DIR}/output/pipeline_$(date +%Y%m%d_%H%M%S).log"

# --- Parse arguments ---
EXTRA_ARGS=""
WITH_GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            EXTRA_ARGS="${EXTRA_ARGS} --quick"
            shift
            ;;
        --phases|-p)
            shift
            PHASES=""
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                PHASES="${PHASES} $1"
                shift
            done
            EXTRA_ARGS="${EXTRA_ARGS} --phases ${PHASES}"
            ;;
        --with-gpu)
            WITH_GPU=true
            shift
            ;;
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --verbose|-v)
            EXTRA_ARGS="${EXTRA_ARGS} --verbose"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# --- Activate environment ---
cd "${PROJECT_DIR}"

if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    echo "Run gcp/setup_vm.sh first."
    exit 1
fi

# --- Ensure output directory exists ---
mkdir -p "${PROJECT_DIR}/output"

# --- Print run configuration ---
echo "============================================="
echo "Pauline Neural Study - Analysis Run"
echo "============================================="
echo "Start time: $(date)"
echo "Config:     ${CONFIG_FILE}"
echo "Output:     ${PROJECT_DIR}/output/"
echo "Log:        ${LOG_FILE}"
echo "Python:     $(python3 --version)"
echo "GPU:        $(command -v nvidia-smi &>/dev/null && echo 'available' || echo 'not available')"
echo "Args:       ${EXTRA_ARGS:-none}"
echo "============================================="
echo ""

# --- Check corpus availability ---
CORPUS_COUNT=$(ls -1 "${PROJECT_DIR}/data/"*.txt 2>/dev/null | wc -l)
if [ "${CORPUS_COUNT}" -eq 0 ]; then
    echo "ERROR: No Greek text files found in ${PROJECT_DIR}/data/"
    echo "The Koine Greek .txt files should be included in the repo."
    exit 1
fi
echo "Greek corpus: ${CORPUS_COUNT} epistle files"
echo ""

# --- If GPU requested, modify config to include VAE phase ---
if [ "${WITH_GPU}" = true ]; then
    echo "GPU mode enabled: including VAE phase"
    # Check if GPU is actually available
    if ! command -v nvidia-smi &>/dev/null; then
        echo "WARNING: --with-gpu specified but no GPU detected!"
        echo "VAE will run on CPU (much slower)."
    fi
    EXTRA_ARGS="${EXTRA_ARGS} --phases corpus bootstrap embeddings cross_epistle combinatorial fractal permutation vae bayesian analysis"
fi

# --- Run the pipeline ---
echo "Starting pipeline..."
echo ""

# Run with logging to both console and file
PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}" \
    python3 -m pauline.pipeline \
    --config "${CONFIG_FILE}" \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Analysis completed successfully!"
else
    echo "Analysis completed with errors (exit code: ${EXIT_CODE})"
fi
echo "============================================="
echo "End time:    $(date)"
echo "Log saved:   ${LOG_FILE}"
echo "Results in:  ${PROJECT_DIR}/output/"
echo ""

# --- List output files ---
echo "Output files:"
ls -lh "${PROJECT_DIR}/output/" 2>/dev/null | tail -n +2 | while read -r line; do
    echo "  ${line}"
done

echo ""
echo "To download results to your local machine:"
echo "  gcloud compute scp <instance>:~/pauline-neural-study/output/* ./results/ --zone <zone>"

exit ${EXIT_CODE}
