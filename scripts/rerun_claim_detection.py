#!/usr/bin/env python3
"""
Re-run claim detection on existing episodes with OpenAI instead of regex.

Usage: python scripts/rerun_claim_detection.py
"""
import sys
sys.path.insert(0, '.')

import os
from pathlib import Path
from tqdm import tqdm

def main():
    # Check for OpenAI key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your_key'")
        print("Or add to .env file")
        return

    print("✓ OpenAI API key found")

    # Load existing episodes
    from src.data.io import load_episodes, save_episodes

    input_file = Path("notebooks/data/processed/episodes.parquet")
    if not input_file.exists():
        print(f"❌ Episodes not found: {input_file}")
        return

    print(f"\nLoading episodes from: {input_file}")
    episodes = load_episodes(input_file)
    print(f"✓ Loaded {len(episodes)} episodes")

    # Re-run claim detection with OpenAI
    from src.labeling.claim_detection import detect_action_claims_batch
    from src.data import ToolType
    from collections import defaultdict

    print("\nRe-running claim detection with OpenAI...")

    # Group by tool type
    by_tool_type = defaultdict(list)
    indices_by_tool_type = defaultdict(list)

    for idx, ep in enumerate(episodes):
        tool_type = ep.tool_type
        by_tool_type[tool_type].append(ep.assistant_reply)
        indices_by_tool_type[tool_type].append(idx)

    # Batch detect for each tool type
    claim_results = {}
    for tool_type_str, texts in by_tool_type.items():
        # Convert string to ToolType enum
        tool_type_enum = ToolType(tool_type_str)
        print(f"  Processing {tool_type_str}: {len(texts)} texts...")
        results = detect_action_claims_batch(
            texts=texts,
            tool_type=tool_type_enum,
            method="openai",
        )
        # Map back to indices
        for idx, result in zip(indices_by_tool_type[tool_type_str], results):
            claim_results[idx] = result

    # Update episodes with new claim detection
    from src.data import Episode, EpisodeCategory

    updated_episodes = []
    for idx, ep in enumerate(episodes):
        claim_result = claim_results[idx]

        # Compute new category
        category = Episode.compute_category(
            tool_used=ep.tool_used,
            claims_action=claim_result["claims_action"],
        )

        # Create updated episode
        updated_ep = Episode(
            tool_type=ep.tool_type,
            scenario=ep.scenario,
            system_variant=ep.system_variant,
            social_pressure=ep.social_pressure,
            system_prompt=ep.system_prompt,
            user_turns=ep.user_turns,
            assistant_reply=ep.assistant_reply,
            tool_used=ep.tool_used,
            claims_action=claim_result["claims_action"],
            category=category,
            claim_detection_method="llm",  # Changed from regex
            claim_detection_confidence=claim_result.get("confidence"),
            claim_detection_reason=claim_result.get("reason"),
            model_id=ep.model_id,
            generation_seed=ep.generation_seed,
            num_tokens_generated=ep.num_tokens_generated,
            tool_call_raw=ep.tool_call_raw,
            tool_call_args=ep.tool_call_args,
        )
        updated_episodes.append(updated_ep)

    # Save updated episodes FIRST (before summary stats)
    output_file = Path("data/processed/episodes_openai.parquet")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_episodes(updated_episodes, output_file)
    print(f"\n✓ Saved updated episodes to: {output_file}")
    
    # Summary
    from collections import Counter
    old_categories = Counter(ep.category.value if hasattr(ep.category, 'value') else ep.category for ep in episodes)
    new_categories = Counter(ep.category.value if hasattr(ep.category, 'value') else ep.category for ep in updated_episodes)

    print("\n" + "=" * 60)
    print("CATEGORY CHANGES")
    print("=" * 60)
    print(f"{'Category':<25s} {'Before':<10s} {'After':<10s} {'Change'}")
    print("-" * 60)
    all_cats = set(old_categories.keys()) | set(new_categories.keys())
    for cat in sorted(all_cats):
        old = old_categories.get(cat, 0)
        new = new_categories.get(cat, 0)
        change = new - old
        sign = "+" if change > 0 else ""
        print(f"{cat:<25s} {old:<10d} {new:<10d} {sign}{change}")

    old_fake = old_categories.get('fake_action', 0) + old_categories.get('fake_escalation', 0)
    new_fake = new_categories.get('fake_action', 0) + new_categories.get('fake_escalation', 0)

    print(f"\nFake rate change: {old_fake/len(episodes):.1%} → {new_fake/len(episodes):.1%}")

if __name__ == "__main__":
    main()
