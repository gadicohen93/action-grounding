#!/usr/bin/env python3
"""
Debug script to help find episode files and resolve path issues.
"""

import sys
from pathlib import Path

def find_episode_files(root_dir=None):
    """Find all episode parquet files in the project."""
    if root_dir is None:
        root_dir = Path(__file__).parent.parent
    else:
        root_dir = Path(root_dir)
    
    episode_files = list(root_dir.rglob("*episodes*.parquet"))
    return episode_files, root_dir

def check_path(path_str, base_dir=None):
    """Check if a path exists and show resolved location."""
    if base_dir:
        path = Path(base_dir) / path_str
    else:
        path = Path(path_str)
    
    exists = path.exists()
    abs_path = path.resolve() if path.exists() else path.absolute()
    
    return {
        "path": path_str,
        "resolved": str(abs_path),
        "exists": exists
    }

def main():
    print("=" * 70)
    print("Episode File Path Debugger")
    print("=" * 70)
    
    # Find all episode files
    episode_files, project_root = find_episode_files()
    
    print(f"\n📁 Project root: {project_root}")
    print(f"\n📊 Current working directory: {Path.cwd()}")
    
    print(f"\n🔍 Found {len(episode_files)} episode file(s):\n")
    for f in sorted(episode_files):
        rel_path = f.relative_to(project_root)
        print(f"  ✓ {rel_path}")
        print(f"    Absolute: {f.resolve()}")
        print()
    
    # Check common problematic paths
    print("\n" + "=" * 70)
    print("Testing Common Paths")
    print("=" * 70 + "\n")
    
    test_paths = [
        "../data/processed/newepisodes.parquet",
        "../data/processed/new_episodes.parquet",
        "../notebooks/data/processed/new_episodes.parquet",
        "data/processed/new_episodes.parquet",
        "notebooks/data/processed/new_episodes.parquet",
    ]
    
    for path_str in test_paths:
        result = check_path(path_str)
        status = "✓ EXISTS" if result["exists"] else "✗ NOT FOUND"
        print(f"{status}: {path_str}")
        print(f"  Resolved to: {result['resolved']}")
        print()
    
    # Suggest fixes
    print("\n" + "=" * 70)
    print("Suggested Fixes")
    print("=" * 70 + "\n")
    
    if episode_files:
        print("Based on available files, try one of these:\n")
        
        # If running from notebooks/
        if Path.cwd().name == "notebooks" or "notebooks" in str(Path.cwd()):
            rel_path = episode_files[0].relative_to(Path.cwd())
            print(f"From notebooks/ directory:")
            print(f"  load_episodes('{rel_path}')")
            print()
        
        # Absolute path
        print("Using absolute path (most reliable):")
        print(f"  from pathlib import Path")
        print(f"  load_episodes(Path('{episode_files[0].resolve()}'))")
        print()
        
        # Relative to project root
        rel_to_root = episode_files[0].relative_to(project_root)
        print(f"Relative to project root:")
        print(f"  load_episodes('{rel_to_root}')")
        print()
    else:
        print("⚠️  No episode files found!")
        print("   Make sure you've generated episodes or synced from RunPod.")

if __name__ == "__main__":
    main()

