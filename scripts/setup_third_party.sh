#!/bin/bash
# Clone and install third-party alignment libraries.
# Run AFTER scripts/env_setup.sh.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

mkdir -p third_party
cd third_party

# platonic-rep (Huh et al., ICML 2024)
if [ ! -d platonic-rep ]; then
    git clone https://github.com/minyoungg/platonic-rep.git
fi
cd platonic-rep
pip install -e . >/dev/null 2>&1
cd ..

# correcting_CKA_alignment (debiased CKA, Murphy ICLR 2024 Re-Align)
if [ ! -d correcting_CKA_alignment ]; then
    git clone https://github.com/Alxmrphi/correcting_CKA_alignment.git
fi

cd ..

# Add top-level script files of platonic-rep to PYTHONPATH via .pth
PTH_FILE=".venv/lib/python3.12/site-packages/platonic-rep.pth"
echo "$REPO_ROOT/third_party/platonic-rep" > "$PTH_FILE"

echo "[setup_third_party] Done."
echo "  platonic-rep: $REPO_ROOT/third_party/platonic-rep"
echo "  correcting_CKA_alignment: $REPO_ROOT/third_party/correcting_CKA_alignment"
echo "  PYTHONPATH pth: $PTH_FILE"

# Verify imports
python - << 'PYEOF'
import measure_alignment
from metrics import AlignmentMetrics
print(f"[verify] platonic-rep metrics: {AlignmentMetrics.SUPPORTED_METRICS if hasattr(AlignmentMetrics, 'SUPPORTED_METRICS') else 'OK'}")
PYEOF
