#!/bin/bash
# Restore episodes_v2.parquet on remote pod from git
# Usage: ./scripts/restore_episodes_v2.sh [POD_IP] [PORT]

set -e

CONFIG_FILE="$HOME/.runpod_connection"

# Try to load from config file if arguments not provided
if [ -z "$1" ] && [ -f "$CONFIG_FILE" ]; then
    echo "Loading connection details from $CONFIG_FILE..."
    source "$CONFIG_FILE"
    POD_IP=${POD_IP:-""}
    PORT=${SSH_PORT:-""}
else
    POD_IP=${1:-""}
    PORT=${2:-""}
fi

# Validate inputs
if [ -z "$POD_IP" ] || [ -z "$PORT" ]; then
    echo "Usage: $0 [POD_IP] [PORT]"
    echo ""
    echo "Options:"
    echo "  1. Provide arguments: $0 <POD_IP> <PORT>"
    echo "  2. Save connection details first: ./scripts/save_pod_connection.sh"
    echo "     Then run: $0"
    echo ""
    echo "Example:"
    echo "  $0 123.45.67.89 12345"
    exit 1
fi

REMOTE_PATH="/workspace/action-grounding"
FILE_PATH="data/processed/episodes_v2.parquet"

echo "=================================================="
echo "Restoring episodes_v2.parquet on remote pod..."
echo "Pod: root@${POD_IP}:${PORT}"
echo "=================================================="
echo ""

# Check if file exists in git history
echo "Checking git history for episodes_v2.parquet..."
COMMIT=$(ssh -p ${PORT} root@${POD_IP} "cd ${REMOTE_PATH} && git log --all --oneline --diff-filter=A -- ${FILE_PATH} | head -1 | cut -d' ' -f1")
if [ -z "$COMMIT" ]; then
    echo "❌ Error: Could not find episodes_v2.parquet in git history"
    exit 1
fi

echo "Found in commit: $COMMIT"
echo ""

# Restore from the commit that added it (c82fc5e) or the latest update (e853045)
echo "Restoring from commit e853045 (fix true action)..."
ssh -p ${PORT} root@${POD_IP} "cd ${REMOTE_PATH} && git checkout e853045 -- ${FILE_PATH}"

# Verify the file was restored
echo ""
echo "Verifying file..."
FILE_SIZE=$(ssh -p ${PORT} root@${POD_IP} "cd ${REMOTE_PATH} && ls -lh ${FILE_PATH} 2>/dev/null | awk '{print \$5}' || echo 'NOT_FOUND'")
if [ "$FILE_SIZE" = "NOT_FOUND" ]; then
    echo "❌ Error: File was not restored"
    exit 1
else
    echo "✓ File restored successfully"
    echo "  Size: $FILE_SIZE"
    echo "  Path: ${REMOTE_PATH}/${FILE_PATH}"
fi

echo ""
echo "=================================================="
echo "✓ Restore complete!"
echo "=================================================="
echo ""
echo "The file should be ~582KB (569KB)."
echo "If you need to commit this change:"
echo "  ssh root@${POD_IP} -p ${PORT}"
echo "  cd ${REMOTE_PATH}"
echo "  git add ${FILE_PATH}"
echo "  git commit -m 'Restore episodes_v2.parquet'"
