#!/usr/bin/env python3
"""
Inspect episodes from data files.

Usage:
    python scripts/inspect_episodes.py [file_path] [--n N] [--category CATEGORY] [--tool TOOL]
    
Examples:
    python scripts/inspect_episodes.py notebooks/data/processed/episodes.parquet --n 5
    python scripts/inspect_episodes.py data/raw/search_episodes_20251221_141706.jsonl --category fake_action
    python scripts/inspect_episodes.py notebooks/data/processed/episodes.parquet --tool search --n 3
"""
import sys
sys.path.insert(0, '.')

import argparse
from pathlib import Path
from collections import Counter

from src.data.io import load_episodes
from src.data import EpisodeCategory, ToolType


def print_episode(ep, idx=None):
    """Pretty print an episode."""
    print("=" * 80)
    if idx is not None:
        print(f"Episode #{idx}")
    print("=" * 80)
    
    # Basic info
    print(f"ID: {ep.id}")
    print(f"Category: {ep.category}")
    print(f"Tool Type: {ep.tool_type}")
    print(f"Scenario: {ep.scenario}")
    print(f"System Variant: {ep.system_variant}")
    print(f"Social Pressure: {ep.social_pressure}")
    print()
    
    # User turns
    print("USER TURNS:")
    for i, turn in enumerate(ep.user_turns, 1):
        print(f"  [{i}] {turn}")
    print()
    
    # Assistant reply
    print("ASSISTANT REPLY:")
    print(f"  {ep.assistant_reply}")
    print()
    
    # Labels
    print("LABELS:")
    print(f"  Tool Used: {ep.tool_used}")
    print(f"  Claims Action: {ep.claims_action}")
    if ep.claim_detection_method:
        print(f"  Detection Method: {ep.claim_detection_method}")
    if ep.claim_detection_confidence is not None:
        print(f"  Confidence: {ep.claim_detection_confidence:.2f}")
    if ep.claim_detection_reason:
        print(f"  Reason: {ep.claim_detection_reason[:200]}...")
    print()
    
    # Tool call info
    if ep.tool_used:
        print("TOOL CALL:")
        if ep.tool_call_raw:
            print(f"  Raw: {ep.tool_call_raw}")
        if ep.tool_call_args:
            print(f"  Args: {ep.tool_call_args}")
        print()
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect episodes from data files")
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to episode file (.parquet, .jsonl, or .json)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of episodes to show (default: 5)"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["true_action", "fake_action", "honest_no_action", "silent_action"],
        help="Filter by category"
    )
    parser.add_argument(
        "--tool",
        type=str,
        choices=["escalate", "search", "sendMessage"],
        help="Filter by tool type"
    )
    parser.add_argument(
        "--fake-only",
        action="store_true",
        help="Show only fake action episodes"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics instead of episodes"
    )
    
    args = parser.parse_args()
    
    # Load episodes
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"Loading episodes from: {file_path}")
    try:
        collection = load_episodes(file_path, validate=True)
    except Exception as e:
        print(f"⚠ Validation failed, trying with conversion: {e}")
        # Try loading with validation disabled (will auto-convert legacy formats)
        collection = load_episodes(file_path, validate=False)
    print(f"✓ Loaded {len(collection)} episodes\n")
    
    # Filter episodes
    episodes = collection.episodes
    
    if args.category:
        category = EpisodeCategory(args.category)
        episodes = [e for e in episodes if e.category == category]
        print(f"Filtered to {len(episodes)} episodes with category={args.category}\n")
    
    if args.tool:
        tool_type = ToolType(args.tool)
        episodes = [e for e in episodes if e.tool_type == tool_type]
        print(f"Filtered to {len(episodes)} episodes with tool={args.tool}\n")
    
    if args.fake_only:
        episodes = [e for e in episodes if e.is_fake()]
        print(f"Filtered to {len(episodes)} fake action episodes\n")
    
    if not episodes:
        print("❌ No episodes match the filters")
        return
    
    # Show summary or episodes
    if args.summary:
        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        categories = Counter(e.category.value if hasattr(e.category, 'value') else e.category for e in episodes)
        tools = Counter(e.tool_type.value if hasattr(e.tool_type, 'value') else e.tool_type for e in episodes)
        
        print("\nBy Category:")
        for cat, count in categories.most_common():
            print(f"  {cat}: {count}")
        
        print("\nBy Tool Type:")
        for tool, count in tools.most_common():
            print(f"  {tool}: {count}")
        
        print(f"\nTotal: {len(episodes)} episodes")
    else:
        # Show episodes
        n = min(args.n, len(episodes))
        print(f"Showing {n} of {len(episodes)} episodes:\n")
        
        for i, ep in enumerate(episodes[:n], 1):
            print_episode(ep, idx=i)
        
        if len(episodes) > n:
            print(f"\n... and {len(episodes) - n} more episodes")


if __name__ == "__main__":
    main()

