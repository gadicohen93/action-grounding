#!/usr/bin/env python
"""
Validation script for RunPod setup.

Run this after setup to verify everything is configured correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 60)
print("RunPod Setup Validation")
print("=" * 60)

checks_passed = 0
checks_failed = 0

# Check 1: CUDA available
print("\n[1/10] Checking CUDA availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        checks_passed += 1
    else:
        print("  ✗ CUDA not available")
        checks_failed += 1
except Exception as e:
    print(f"  ✗ Error: {e}")
    checks_failed += 1

# Check 2: Transformers installed
print("\n[2/10] Checking Transformers...")
try:
    import transformers
    print(f"  ✓ Transformers {transformers.__version__}")
    checks_passed += 1
except Exception as e:
    print(f"  ✗ Error: {e}")
    checks_failed += 1

# Check 3: Config loads
print("\n[3/10] Checking config.yaml...")
try:
    from src.config import get_config
    config = get_config()
    print(f"  ✓ Config loaded")
    print(f"    Model: {config.model.id}")
    print(f"    Quantization: {config.model.quantization}")
    checks_passed += 1
except Exception as e:
    print(f"  ✗ Error: {e}")
    checks_failed += 1

# Check 4: Secrets (API keys)
print("\n[4/10] Checking API keys...")
try:
    from src.config import get_secrets
    secrets = get_secrets()

    if secrets.hf_token:
        print(f"  ✓ HuggingFace token: {secrets.hf_token[:10]}...")
    else:
        print(f"  ✗ HuggingFace token not set")
        checks_failed += 1

    if secrets.openai_api_key:
        print(f"  ✓ OpenAI key: {secrets.openai_api_key[:10]}...")
    else:
        print(f"  ✗ OpenAI key not set")
        checks_failed += 1

    if secrets.hf_token and secrets.openai_api_key:
        checks_passed += 1
except Exception as e:
    print(f"  ✗ Error: {e}")
    checks_failed += 1

# Check 5: Backend loads
print("\n[5/10] Checking backend loading...")
try:
    from src.backends import PyTorchBackend
    print("  ✓ PyTorchBackend imported")
    checks_passed += 1
except Exception as e:
    print(f"  ✗ Error: {e}")
    checks_failed += 1

# Check 6: Data schemas
print("\n[6/10] Checking data schemas...")
try:
    from src.data import Episode, ActivationDataset
    print("  ✓ Data schemas imported")
    checks_passed += 1
except Exception as e:
    print(f"  ✗ Error: {e}")
    checks_failed += 1

# Check 7: Generation module
print("\n[7/10] Checking generation module...")
try:
    from src.generation import get_all_conditions
    conditions = get_all_conditions()
    print(f"  ✓ Generation module OK ({len(conditions)} conditions)")
    checks_passed += 1
except Exception as e:
    print(f"  ✗ Error: {e}")
    checks_failed += 1

# Check 8: Analysis module
print("\n[8/10] Checking analysis module...")
try:
    from src.analysis import train_probe, bootstrap_ci
    print("  ✓ Analysis module OK")
    checks_passed += 1
except Exception as e:
    print(f"  ✗ Error: {e}")
    checks_failed += 1

# Check 9: Intervention module
print("\n[9/10] Checking intervention module...")
try:
    from src.intervention import SteeringExperiment
    print("  ✓ Intervention module OK")
    checks_passed += 1
except Exception as e:
    print(f"  ✗ Error: {e}")
    checks_failed += 1

# Check 10: Data directories
print("\n[10/10] Checking data directories...")
required_dirs = [
    "data/raw",
    "data/processed",
    "figures",
    "logs",
]
for dir_path in required_dirs:
    path = Path(dir_path)
    if path.exists():
        print(f"  ✓ {dir_path}/")
    else:
        print(f"  ✗ {dir_path}/ not found (creating...)")
        path.mkdir(parents=True, exist_ok=True)
checks_passed += 1

# Summary
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"Passed: {checks_passed}/10")
print(f"Failed: {checks_failed}/10")

if checks_failed == 0:
    print("\n✓ ALL CHECKS PASSED - Ready to run experiments!")
    print("\nNext steps:")
    print("  1. Start Jupyter: tmux new -s jupyter")
    print("  2. In tmux: source venv/bin/activate && jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root")
    print("  3. Open notebooks/01_behavioral_phenomenon.ipynb")
    sys.exit(0)
else:
    print("\n✗ VALIDATION FAILED - Fix errors above before running")
    print("\nCommon fixes:")
    print("  - API keys: nano .env (add HF_TOKEN and OPENAI_API_KEY)")
    print("  - Dependencies: pip install -r requirements.txt")
    print("  - CUDA: Verify you're on a GPU pod (nvidia-smi)")
    sys.exit(1)
