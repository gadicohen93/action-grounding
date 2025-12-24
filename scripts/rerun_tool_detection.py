#!/usr/bin/env python3
"""
Re-run tool detection to add tool_used_any field.

Usage: python scripts/rerun_tool_detection.py [--dry-run]
"""
import sys
sys.path.insert(0, '.')

import argparse
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/episodes_openai.parquet")
    parser.add_argument("--output", default="data/processed/episodes_v2.parquet")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from src.data.io import load_episodes, save_episodes
    from src.data import Episode, ToolType
    from src.labeling.tool_detection import detect_tool_call_extended

    # Load
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Episodes not found: {input_path}")
        return

    print(f"Loading episodes from: {input_path}")
    episodes = load_episodes(input_path)
    print(f"✓ Loaded {len(episodes)} episodes")

    # Process
    updated = []
    changes = Counter()

    for ep in episodes:
        tool_type = ToolType(ep.tool_type)
        result = detect_tool_call_extended(ep.assistant_reply, tool_type)

        new_category = Episode.compute_category_v2(
            result["tool_used"], result["tool_used_any"], ep.claims_action
        )

        old_cat = ep.category
        if old_cat != new_category:
            changes[f"{old_cat} -> {new_category}"] += 1

        # Preserve all existing fields, update only the new ones
        ep_dict = ep.model_dump()
        ep_dict["tool_used_any"] = result["tool_used_any"]
        ep_dict["wrong_tool_name"] = result.get("wrong_tool_name")
        ep_dict["category"] = new_category

        updated.append(Episode(**ep_dict))

    # Report
    print("\n" + "=" * 60)
    print("CATEGORY CHANGES")
    print("=" * 60)
    if changes:
        for change, count in changes.most_common():
            print(f"  {change}: {count}")
    else:
        print("  No category changes")

    # Category distribution
    from collections import Counter as CounterType
    old_categories = CounterType(ep.category.value if hasattr(ep.category, 'value') else ep.category for ep in episodes)
    new_categories = CounterType(ep.category.value if hasattr(ep.category, 'value') else ep.category for ep in updated)

    print("\n" + "=" * 60)
    print("CATEGORY DISTRIBUTION")
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

    # Wrong tool stats
    wrong_tool_count = sum(1 for ep in updated if ep.wrong_tool_name is not None)
    if wrong_tool_count > 0:
        wrong_tools = CounterType(ep.wrong_tool_name for ep in updated if ep.wrong_tool_name is not None)
        print(f"\nWrong tools called: {wrong_tool_count}")
        for tool, count in wrong_tools.most_common():
            print(f"  {tool}: {count}")

    if args.dry_run:
        print("\n[DRY RUN] No changes saved")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_episodes(updated, output_path)
        print(f"\n✓ Saved updated episodes to: {output_path}")

if __name__ == "__main__":
    main()

