"""
Example: Using checkpointing with steering experiments

This demonstrates how to use the checkpoint feature to resume interrupted experiments.
"""

from pathlib import Path
from src.config import get_config
from src.data.io import load_episodes
from src.analysis.probes import load_probe, get_probe_direction
from src.intervention.steering import run_steering_experiment

config = get_config()

# Load probe and episodes (example)
reality_probe = load_probe(config.data.processed_dir / "reality_probe.pkl")
probe_direction = get_probe_direction(reality_probe)

episodes_collection = load_episodes(config.data.processed_dir / "episodes_v2.parquet")
fake_episodes = episodes_collection.get_fake_episodes()
fake_sample = fake_episodes[:50]  # Sample

# Set checkpoint path - progress will be saved here incrementally
checkpoint_path = config.data.processed_dir / "fake_steering_checkpoint.json"

# Run with checkpointing
# If checkpoint exists, it will automatically resume from where it left off
fake_steering_results = run_steering_experiment(
    probe_direction=probe_direction,
    episodes=fake_sample,
    alphas=config.steering.alphas,
    model_id=config.model.id,
    target_layer=config.steering.target_layer,
    verbose=True,
    checkpoint_path=checkpoint_path,  # <-- Add this parameter
)

print(f"\nCompleted {len(fake_steering_results)} steering experiments")
print(f"Checkpoint saved to: {checkpoint_path}")

