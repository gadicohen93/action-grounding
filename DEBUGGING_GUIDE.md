# Debugging FileNotFoundError for Episode Files

## The Problem

You're trying to load:
```python
episodes_collection = load_episodes("../data/processed/newepisodes.parquet")
```

But getting:
```
FileNotFoundError: Episode file not found: ../data/processed/newepisodes.parquet
```

## Issues Found

### 1. **Filename Typo**
The file is named `new_episodes.parquet` (with underscore), not `newepisodes.parquet`.

### 2. **Path Resolution**
The relative path `../data/processed/` depends on your current working directory. If you're running code from `notebooks/`, the path should be different.

## Available Episode Files

Based on the codebase, here are the episode files that exist:

### Main data directory:
- `data/processed/episodes_v2.parquet`
- `data/processed/episodes_openai.parquet`

### Notebooks data directory:
- `notebooks/data/processed/new_episodes.parquet` ✅ (This is likely what you want)
- `notebooks/data/processed/episodes.parquet`

## Solutions

### Option 1: Fix the filename (if running from notebooks/)
```python
# If running from notebooks/ directory
episodes_collection = load_episodes("../notebooks/data/processed/new_episodes.parquet")
```

### Option 2: Use absolute path (recommended)
```python
from pathlib import Path

# Use absolute path from project root
episodes_collection = load_episodes(
    Path(__file__).parent.parent / "notebooks" / "data" / "processed" / "new_episodes.parquet"
)
```

### Option 3: Use existing files in main data directory
```python
# Use episodes_v2.parquet from main data directory
episodes_collection = load_episodes("../data/processed/episodes_v2.parquet")
```

## Debugging Steps

### Step 1: Check your current working directory
```python
import os
from pathlib import Path

print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {Path(__file__).parent if '__file__' in globals() else 'N/A'}")
```

### Step 2: Check if file exists
```python
from pathlib import Path

# Try different paths
paths_to_check = [
    "../data/processed/newepisodes.parquet",
    "../data/processed/new_episodes.parquet",
    "../notebooks/data/processed/new_episodes.parquet",
    "data/processed/new_episodes.parquet",
    "notebooks/data/processed/new_episodes.parquet",
]

for path_str in paths_to_check:
    path = Path(path_str)
    exists = path.exists()
    abs_path = path.resolve()
    print(f"{'✓' if exists else '✗'} {path_str}")
    print(f"  Resolved to: {abs_path}")
    print(f"  Exists: {exists}\n")
```

### Step 3: List all episode files
```python
from pathlib import Path

# Find all episode parquet files
project_root = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
episode_files = list(project_root.rglob("*episodes*.parquet"))

print("Found episode files:")
for f in episode_files:
    print(f"  {f.relative_to(project_root)}")
```

### Step 4: Use the inspect script
```bash
# List available files
ls -la data/processed/*.parquet
ls -la notebooks/data/processed/*.parquet

# Inspect a specific file
python scripts/inspect_episodes.py notebooks/data/processed/new_episodes.parquet --summary
```

## Quick Fix

If you're in a Jupyter notebook in the `notebooks/` directory, use:

```python
from pathlib import Path
from src.data import load_episodes

# Option A: Relative to notebooks/
episodes_collection = load_episodes("../notebooks/data/processed/new_episodes.parquet")

# Option B: Absolute path (more reliable)
project_root = Path().resolve().parent  # Go up from notebooks/ to project root
episodes_collection = load_episodes(
    project_root / "notebooks" / "data" / "processed" / "new_episodes.parquet"
)

# Option C: Use main data directory
episodes_collection = load_episodes("../data/processed/episodes_v2.parquet")
```

