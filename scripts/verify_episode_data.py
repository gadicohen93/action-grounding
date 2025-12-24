#!/usr/bin/env python3
"""
Verify episode data integrity.

Usage: python scripts/verify_episode_data.py
"""
import sys
sys.path.insert(0, '.')

import json
from pathlib import Path

def inspect_episode(episode_file, episode_idx=0):
    """Inspect a single episode in detail."""

    # Load from JSONL
    print(f"Loading from: {episode_file}")
    with open(episode_file) as f:
        episodes_raw = [json.loads(line) for line in f]

    print(f"Total episodes: {len(episodes_raw)}\n")

    # Get specific episode
    ep = episodes_raw[episode_idx]

    print("=" * 70)
    print(f"EPISODE {episode_idx} INSPECTION")
    print("=" * 70)

    # Check keys
    print(f"\nKeys: {list(ep.keys())}\n")

    # Check assistant_reply
    reply = ep.get('assistant_reply') or ep.get('reply')
    print(f"Assistant reply field name: {'assistant_reply' if 'assistant_reply' in ep else 'reply'}")
    print(f"Reply length: {len(reply) if reply else 0} chars")
    print(f"\nFirst 500 chars of reply:")
    print("-" * 70)
    print(reply[:500] if reply else "NO REPLY FOUND")
    print("-" * 70)

    if reply and len(reply) > 500:
        print(f"\n[Reply continues for {len(reply)-500} more chars...]")
        print(f"\nLast 200 chars:")
        print("-" * 70)
        print(reply[-200:])
        print("-" * 70)

    # Check tool usage
    print(f"\nTool used: {ep.get('tool_used')}")
    print(f"Claims action: {ep.get('claims_action')}")
    print(f"Category: {ep.get('category')}")

    # Check for tool call in reply
    has_call = '<<CALL' in reply if reply else False
    print(f"\nContains <<CALL: {has_call}")

    if has_call:
        # Extract the tool call
        start_idx = reply.find('<<CALL')
        end_idx = reply.find('>>', start_idx + 6)
        if end_idx != -1:
            tool_call = reply[start_idx:end_idx+2]
            print(f"Tool call found: {tool_call}")

    # Verify consistency
    print(f"\n{'='*70}")
    print("CONSISTENCY CHECK")
    print("=" * 70)

    if ep.get('tool_used') and not has_call:
        print("⚠️  WARNING: tool_used=True but no <<CALL found in reply")
    elif not ep.get('tool_used') and has_call:
        print("⚠️  WARNING: tool_used=False but <<CALL found in reply")
    else:
        print("✓ tool_used matches <<CALL presence")

    category = ep.get('category', '')
    if 'fake' in category and ep.get('tool_used'):
        print("⚠️  WARNING: Category is 'fake' but tool_used=True (inconsistent)")
    elif 'fake' in category and not has_call and ep.get('claims_action'):
        print("✓ Fake category consistent: no tool call but claims action")

    return ep

def check_all_episodes(episode_file):
    """Quick check of all episodes."""
    with open(episode_file) as f:
        episodes = [json.loads(line) for line in f]

    print(f"\n{'='*70}")
    print(f"CHECKING ALL {len(episodes)} EPISODES")
    print("=" * 70)

    issues = []

    for i, ep in enumerate(episodes):
        reply = ep.get('assistant_reply') or ep.get('reply')

        # Check for common issues
        if not reply:
            issues.append(f"Episode {i}: No reply field")
        elif len(reply) < 10:
            issues.append(f"Episode {i}: Reply too short ({len(reply)} chars)")
        elif reply.startswith('[/INST]') or reply.startswith('[INST]'):
            issues.append(f"Episode {i}: Reply starts with prompt tokens")

        # Check consistency
        has_call = '<<CALL' in reply if reply else False
        if ep.get('tool_used') and not has_call:
            issues.append(f"Episode {i}: tool_used=True but no <<CALL")
        elif not ep.get('tool_used') and has_call:
            issues.append(f"Episode {i}: tool_used=False but has <<CALL")

    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues)-10} more")
    else:
        print("\n✓ All episodes look good!")

    return issues

if __name__ == "__main__":
    # Inspect first episode
    episode_file = Path("data/processed/episodes.parquet")

    # If parquet, need to convert
    if episode_file.suffix == '.parquet':
        print("Loading from Parquet...")
        sys.path.insert(0, '.')
        from src.data.io import load_episodes
        episodes = load_episodes(episode_file)

        print(f"Loaded {len(episodes)} episodes from Parquet")
        print("\nFirst episode:")
        ep = episodes[0]
        print(f"  assistant_reply length: {len(ep.assistant_reply)} chars")
        print(f"  First 300 chars: {ep.assistant_reply[:300]}")
        print(f"  tool_used: {ep.tool_used}")
        print(f"  claims_action: {ep.claims_action}")
        print(f"  category: {ep.category}")

        # Check for issues
        has_call = '<<CALL' in ep.assistant_reply
        print(f"  Contains <<CALL: {has_call}")

        if ep.tool_used and not has_call:
            print("  ⚠️  WARNING: tool_used=True but no <<CALL found")
        elif not ep.tool_used and has_call:
            print("  ⚠️  WARNING: tool_used=False but <<CALL found")
        else:
            print("  ✓ Consistency check passed")
    else:
        # JSONL format
        inspect_episode(episode_file, episode_idx=0)
        check_all_episodes(episode_file)
