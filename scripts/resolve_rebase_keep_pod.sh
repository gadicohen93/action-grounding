#!/bin/bash
# Resolve git rebase conflicts by keeping pod versions (--theirs)

set -e

echo "Resolving rebase conflicts by keeping pod versions..."

# Check if we're in a rebase
if [ ! -d ".git/rebase-merge" ] && [ ! -d ".git/rebase-apply" ]; then
    echo "Error: No rebase in progress"
    exit 1
fi

# Keep pod versions for all conflicted notebook files
echo "Keeping pod versions for notebook files..."
git checkout --theirs notebooks/01_behavioral_phenomenon.ipynb 2>/dev/null || echo "  Skipping 01_behavioral_phenomenon.ipynb (no conflict)"
git checkout --theirs notebooks/02_mechanistic_probes.ipynb 2>/dev/null || echo "  Skipping 02_mechanistic_probes.ipynb (no conflict)"
git checkout --theirs notebooks/03_analysis_existing_data.ipynb 2>/dev/null || echo "  Skipping 03_analysis_existing_data.ipynb (no conflict)"
git checkout --theirs notebooks/04_causal_intervention.ipynb 2>/dev/null || echo "  Skipping 04_causal_intervention.ipynb (no conflict)"

# Mark all notebook files as resolved
echo "Marking notebooks as resolved..."
git add notebooks/*.ipynb

echo ""
echo "Resolved! Run 'git rebase --continue' to proceed."
echo ""
echo "Or run this script with --continue flag to automatically continue:"
echo "  $0 --continue"

if [ "$1" == "--continue" ]; then
    echo ""
    echo "Continuing rebase..."
    git rebase --continue
fi

