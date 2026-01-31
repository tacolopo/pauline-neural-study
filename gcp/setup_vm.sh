#!/usr/bin/env bash
# =============================================================================
# Pauline Neural Study - GCP VM Setup Script
# =============================================================================
#
# This script sets up a Google Cloud Platform VM for running the Pauline
# Neural Study pipeline. It installs all dependencies, configures the
# environment, and prepares the data.
#
# Usage:
#   1. Create a GCP VM (recommended: e2-standard-4 or n1-standard-4)
#      For VAE phase: use a GPU instance (e.g., n1-standard-4 + NVIDIA T4)
#
#   2. SSH into the VM:
#      gcloud compute ssh <instance-name> --zone <zone>
#
#   3. Clone the repo and run this script:
#      git clone <repo-url> ~/pauline-neural-study
#      cd ~/pauline-neural-study
#      chmod +x gcp/setup_vm.sh
#      bash gcp/setup_vm.sh
#
#   4. Run the analysis:
#      bash gcp/run_analysis.sh
#
# Recommended VM Specs:
#   - CPU-only (Phases 1-7, 9-10): e2-standard-4 (4 vCPUs, 16 GB RAM)
#   - With GPU (Phase 8 - VAE):    n1-standard-4 + NVIDIA T4
#   - Disk: 20 GB SSD (default is fine)
#   - OS: Ubuntu 22.04 LTS or Debian 12
#
# Estimated costs (as of 2026):
#   - e2-standard-4: ~$0.13/hr
#   - n1-standard-4 + T4: ~$0.55/hr
#   - Full pipeline run: ~1-3 hours depending on config
#
# =============================================================================

set -euo pipefail

echo "============================================="
echo "Pauline Neural Study - GCP VM Setup"
echo "============================================="
echo ""

# --- Configuration ---
PROJECT_DIR="${HOME}/pauline-neural-study"
VENV_DIR="${PROJECT_DIR}/.venv"

# --- Detect OS ---
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_NAME="${ID}"
    OS_VERSION="${VERSION_ID}"
else
    echo "WARNING: Cannot detect OS. Assuming Debian/Ubuntu."
    OS_NAME="debian"
fi

echo "Detected OS: ${OS_NAME} ${OS_VERSION:-unknown}"
echo "Project directory: ${PROJECT_DIR}"
echo ""

# --- Detect Python version ---
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3.10")
echo "Detected Python: ${PYTHON_VERSION}"

# --- System packages ---
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 \
    python3-venv \
    "python${PYTHON_VERSION}-venv" \
    python3-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    pkg-config \
    libhdf5-dev \
    2>/dev/null || true

echo "  Python version: $(python3 --version)"
echo ""

# --- Create virtual environment ---
echo "[2/6] Creating Python virtual environment..."
if [ -d "${VENV_DIR}" ]; then
    echo "  Virtual environment already exists, reusing."
else
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel -q

echo "  Virtual environment: ${VENV_DIR}"
echo "  pip version: $(pip --version)"
echo ""

# --- Install Python dependencies ---
echo "[3/6] Installing Python dependencies..."
cd "${PROJECT_DIR}"

if [ -f requirements.txt ]; then
    pip install -r requirements.txt -q
else
    echo "  WARNING: requirements.txt not found, installing core packages..."
    pip install -q \
        numpy \
        scipy \
        gensim \
        nltk \
        scikit-learn \
        matplotlib \
        seaborn \
        pyyaml \
        requests \
        tqdm
fi

# Install the project itself
pip install -e . -q 2>/dev/null || echo "  Note: pip install -e . skipped (no setup.py/pyproject.toml)"

echo "  Installed packages: $(pip list --format=columns | wc -l) packages"
echo ""

# --- Check for GPU (optional, for VAE phase) ---
echo "[4/6] Checking GPU availability..."
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    echo ""
    echo "  Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null || \
        pip install torch -q
else
    echo "  No GPU detected. VAE phase will use CPU (slower)."
    echo "  Installing PyTorch (CPU only)..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu -q 2>/dev/null || \
        pip install torch -q
fi
echo ""

# --- Verify Greek corpus data ---
echo "[5/6] Verifying Koine Greek corpus..."
mkdir -p "${PROJECT_DIR}/output"

CORPUS_COUNT=$(ls -1 "${PROJECT_DIR}/data/"*.txt 2>/dev/null | wc -l)
if [ "${CORPUS_COUNT}" -eq 0 ]; then
    echo "  ERROR: No Greek text files found in ${PROJECT_DIR}/data/"
    echo "  The 14 Pauline epistle .txt files should be included in the repo."
    exit 1
fi
echo "  Found ${CORPUS_COUNT} Greek text files"

# --- Validate corpus loads correctly ---
echo "[6/6] Validating corpus loading..."
python3 -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/src')
from pauline.corpus.loader import PaulineCorpus
corpus = PaulineCorpus.from_text_files('${PROJECT_DIR}/data', undisputed_only=False)
summary = corpus.summary()
print(f'  Corpus loaded: {summary[\"total_words\"]} words, {summary[\"vocabulary_size\"]} unique')
print(f'  Epistles ({summary[\"epistles\"]}): {\", \".join(summary[\"epistle_names\"])}')
print(f'  Language: {\"Greek\" if summary[\"is_greek\"] else \"English\"}')
" || echo "  WARNING: Corpus validation failed. Check data/ directory."

echo ""
echo "============================================="
echo "Setup complete!"
echo "============================================="
echo ""
echo "To run the analysis:"
echo "  cd ${PROJECT_DIR}"
echo "  bash gcp/run_analysis.sh"
echo ""
echo "Or manually:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python -m pauline.pipeline --config configs/default.yaml"
echo ""
echo "For a quick test:"
echo "  python -m pauline.pipeline --quick"
echo ""
