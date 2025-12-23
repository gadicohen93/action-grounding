#!/bin/bash
# Run all notebooks sequentially on RunPod
# This converts notebooks to Python scripts and executes them

set -e

echo "=================================================="
echo "Running All Notebooks Sequentially"
echo "=================================================="

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Convert notebooks to Python scripts
echo ""
echo "[Step 1/5] Converting notebooks to Python scripts..."
jupyter nbconvert --to python notebooks/01_behavioral_phenomenon.ipynb
jupyter nbconvert --to python notebooks/02_mechanistic_probes.ipynb
jupyter nbconvert --to python notebooks/03_generalization.ipynb
jupyter nbconvert --to python notebooks/04_causal_intervention.ipynb
echo "✓ Notebooks converted"

# Run Notebook 01
echo ""
echo "=================================================="
echo "[Step 2/5] Running Notebook 01: Behavioral Phenomenon"
echo "Expected time: 2-4 hours"
echo "=================================================="
python notebooks/01_behavioral_phenomenon.py 2>&1 | tee logs/nb01_$(date +%Y%m%d_%H%M%S).log
echo "✓ Notebook 01 complete"

# Verify output
if [ ! -f "data/processed/episodes.parquet" ]; then
    echo "❌ ERROR: episodes.parquet not found!"
    exit 1
fi
echo "✓ Verified: episodes.parquet exists"

# Run Notebook 02
echo ""
echo "=================================================="
echo "[Step 3/5] Running Notebook 02: Mechanistic Probes"
echo "Expected time: 1-2 hours"
echo "=================================================="
python notebooks/02_mechanistic_probes.py 2>&1 | tee logs/nb02_$(date +%Y%m%d_%H%M%S).log
echo "✓ Notebook 02 complete"

# Verify outputs
if [ ! -f "data/processed/activations.parquet" ]; then
    echo "❌ ERROR: activations.parquet not found!"
    exit 1
fi
if [ ! -f "data/processed/reality_probe.pkl" ]; then
    echo "❌ ERROR: reality_probe.pkl not found!"
    exit 1
fi
echo "✓ Verified: activations and probes exist"

# Run Notebook 03
echo ""
echo "=================================================="
echo "[Step 4/5] Running Notebook 03: Generalization"
echo "Expected time: 30 minutes"
echo "=================================================="
python notebooks/03_generalization.py 2>&1 | tee logs/nb03_$(date +%Y%m%d_%H%M%S).log
echo "✓ Notebook 03 complete"

# Run Notebook 04
echo ""
echo "=================================================="
echo "[Step 5/5] Running Notebook 04: Causal Intervention"
echo "Expected time: 2 hours"
echo "=================================================="
python notebooks/04_causal_intervention.py 2>&1 | tee logs/nb04_$(date +%Y%m%d_%H%M%S).log
echo "✓ Notebook 04 complete"

# Summary
echo ""
echo "=================================================="
echo "✓ ALL NOTEBOOKS COMPLETE!"
echo "=================================================="
echo ""
echo "Results:"
ls -lh data/processed/
echo ""
echo "Figures:"
ls -lh figures/
echo ""
echo "Logs saved to: logs/"
echo ""
echo "Next: Sync results back to your Mac"
echo "  scp -P <PORT> -r root@<POD_IP>:/workspace/action-grounding/figures/* ~/figures/"
echo ""
echo "=================================================="
