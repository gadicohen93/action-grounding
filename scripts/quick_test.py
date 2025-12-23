#!/usr/bin/env python
"""
Quick test script for RunPod setup.

Runs a minimal version of the full pipeline to verify everything works.
Generates 5 episodes, extracts activations, trains a probe.

Total time: ~5-10 minutes
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging
from src.utils.logging import setup_logging
from src.config import get_config
from src.generation import generate_batch, get_all_conditions
from src.generation.prompts import ToolType
from src.extraction import extract_activations_batch
from src.analysis.probes import train_and_evaluate

setup_logging(level="INFO")
logger = logging.getLogger(__name__)

print("=" * 60)
print("QUICK TEST: Minimal Pipeline Run")
print("=" * 60)

# Load config
config = get_config()
logger.info(f"Model: {config.model.id}")
logger.info(f"Quantization: {config.model.quantization}")

# Test 1: Generate small batch
print("\n[Test 1/4] Generating 5 test episodes...")
print("  (This tests: model loading, generation, labeling)")

conditions = get_all_conditions(
    tool_types=[ToolType.ESCALATE],  # Just one tool
)[:1]  # Just one condition

test_episodes = generate_batch(
    conditions=conditions,
    n_per_condition=5,
    model_id=config.model.id,
    labeling_method="openai",
    verbose=True,
)

logger.info(f"✓ Generated {len(test_episodes)} episodes")

# Check results
fake_count = sum(1 for e in test_episodes if e.is_fake())
logger.info(f"  Fake episodes: {fake_count}/{len(test_episodes)}")

# Test 2: Extract activations
print("\n[Test 2/4] Extracting activations...")
print("  (This tests: activation extraction, position finding)")

test_dataset = extract_activations_batch(
    episodes=test_episodes,
    positions=["first_assistant", "mid_response"],
    layers=[16],  # Just one layer
    model_id=config.model.id,
    verbose=True,
)

logger.info(f"✓ Extracted {len(test_dataset)} activation samples")
logger.info(f"  Shape: {test_dataset.activations.shape}")

# Test 3: Train probe
print("\n[Test 3/4] Training probe...")
print("  (This tests: probe training, evaluation)")

if len(test_dataset) >= 8:
    probe, train_metrics, test_metrics = train_and_evaluate(
        test_dataset,
        label_type="reality",
        test_size=0.2,
        random_state=42,
    )

    logger.info(f"✓ Probe trained")
    logger.info(f"  Test accuracy: {test_metrics.accuracy:.1%}")
else:
    logger.warning("Not enough samples for probe training (need 8+)")

# Test 4: Check outputs
print("\n[Test 4/4] Checking module imports...")

modules_to_check = [
    "src.backends",
    "src.data",
    "src.generation",
    "src.labeling",
    "src.extraction",
    "src.analysis",
    "src.intervention",
    "src.utils",
]

all_imported = True
for module in modules_to_check:
    try:
        __import__(module)
        logger.info(f"  ✓ {module}")
    except Exception as e:
        logger.error(f"  ✗ {module}: {e}")
        all_imported = False

# Summary
print("\n" + "=" * 60)
print("QUICK TEST SUMMARY")
print("=" * 60)

if all_imported and len(test_episodes) > 0 and len(test_dataset) > 0:
    print("✓ ALL TESTS PASSED!")
    print("\nYou're ready to run the full pipeline:")
    print("  Option 1: Run notebooks in Jupyter")
    print("  Option 2: Run automated script: ./scripts/run_all_notebooks.sh")
    print("\nEstimated full runtime: ~8-12 hours")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED")
    print("\nCheck error messages above and fix before running full pipeline")
    sys.exit(1)
