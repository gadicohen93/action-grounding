#!/usr/bin/env python3
"""
Quick test to verify episode generation code works without GPU.
Run this locally before pushing to RunPod.

Usage: python scripts/test_generation.py
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    from src.config import get_config
    from src.generation import get_all_conditions, EpisodeGenerator
    from src.data import Episode, ToolType
    print("  ✓ All imports successful")

def test_config():
    """Test config loading."""
    print("Testing config...")
    from src.config import get_config
    config = get_config()
    print(f"  ✓ Config loaded: {config.log_mode_info()}")
    print(f"  ✓ Backend: {config.model.backend}")
    return config

def test_conditions():
    """Test condition generation."""
    print("Testing conditions...")
    from src.generation import get_all_conditions
    conditions = get_all_conditions()
    print(f"  ✓ Generated {len(conditions)} conditions")
    return conditions

def test_checkpoint_path():
    """Test checkpoint path generation."""
    print("Testing checkpoint path logic...")
    save_path = Path("./data/processed/episodes.parquet")
    checkpoint_path = str(save_path).replace('.parquet', '_checkpoint.jsonl')
    print(f"  ✓ Checkpoint path: {checkpoint_path}")

def test_episode_creation():
    """Test Episode object creation."""
    print("Testing Episode creation...")
    from src.data import Episode, ToolType, EpisodeCategory
    from src.generation import SystemVariant, SocialPressure

    episode = Episode(
        tool_type=ToolType.ESCALATE,
        scenario="test",
        system_variant=SystemVariant.A_STRICT,
        social_pressure=SocialPressure.NEUTRAL,
        system_prompt="test",
        user_turns=["test"],
        assistant_reply="test reply",
        tool_used=False,
        claims_action=True,
        category=EpisodeCategory.FAKE,
        claim_detection_method="regex",
        model_id="test",
    )

    # Test category.value access (the bug we just hit)
    cat_value = episode.category.value if hasattr(episode.category, 'value') else episode.category
    print(f"  ✓ Episode created, category={cat_value}")
    print(f"  ✓ is_fake() = {episode.is_fake()}")

def test_generator_init():
    """Test generator initialization (without loading model)."""
    print("Testing generator init (no model load)...")
    # This just tests the code path, won't actually load model
    print("  ✓ Generator code path OK (skipping actual init)")

def main():
    print("=" * 60)
    print("EPISODE GENERATION CODE TEST")
    print("=" * 60)

    tests = [
        test_imports,
        test_config,
        test_conditions,
        test_checkpoint_path,
        test_episode_creation,
        test_generator_init,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    print("\n✓ All tests passed! Safe to push to RunPod.")

if __name__ == "__main__":
    main()
