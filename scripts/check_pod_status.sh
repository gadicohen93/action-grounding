#!/bin/bash
# Check what's on the RunPod pod
# Usage: ./scripts/check_pod_status.sh [POD_IP] [PORT]

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
    echo "  $0 69.30.85.120 22130"
    exit 1
fi

REMOTE_PATH="/workspace/action-grounding"

echo "=================================================="
echo "Checking RunPod Pod Status"
echo "Pod: root@${POD_IP}:${PORT}"
echo "=================================================="
echo ""

# Check if we can connect
echo "[1/7] Testing SSH connection..."
SSH_CMD="ssh -p ${PORT} -o ConnectTimeout=5"

# Try with common SSH key locations if default doesn't work
if ssh -p ${PORT} -o ConnectTimeout=5 root@${POD_IP} "echo 'Connection OK'" 2>/dev/null; then
    echo "✓ SSH connection successful"
elif [ -f ~/.ssh/id_ed25519 ] && ${SSH_CMD} -i ~/.ssh/id_ed25519 root@${POD_IP} "echo 'Connection OK'" 2>/dev/null; then
    echo "✓ SSH connection successful (using ~/.ssh/id_ed25519)"
    SSH_CMD="${SSH_CMD} -i ~/.ssh/id_ed25519"
elif [ -f ~/.ssh/id_rsa ] && ${SSH_CMD} -i ~/.ssh/id_rsa root@${POD_IP} "echo 'Connection OK'" 2>/dev/null; then
    echo "✓ SSH connection successful (using ~/.ssh/id_rsa)"
    SSH_CMD="${SSH_CMD} -i ~/.ssh/id_rsa"
else
    echo "❌ Cannot connect to pod. Check IP and port."
    echo ""
    echo "Try manually:"
    echo "  ssh root@${POD_IP} -p ${PORT}"
    echo "  # Or with SSH key:"
    echo "  ssh root@${POD_IP} -p ${PORT} -i ~/.ssh/id_ed25519"
    exit 1
fi

# Check workspace directory
echo ""
echo "[2/7] Checking workspace directory..."
if ${SSH_CMD} root@${POD_IP} "test -d /workspace && echo 'exists' || echo 'missing'" 2>/dev/null | grep -q "exists"; then
    echo "✓ /workspace directory exists"
    WORKSPACE_CONTENTS=$(${SSH_CMD} root@${POD_IP} "ls -la /workspace 2>/dev/null | head -10" 2>/dev/null)
    echo "  Contents:"
    echo "$WORKSPACE_CONTENTS" | sed 's/^/    /'
else
    echo "⚠ /workspace directory not found (checking /root instead)"
fi

# Check if repo exists
echo ""
echo "[3/7] Checking if repository exists..."
if ${SSH_CMD} root@${POD_IP} "test -d ${REMOTE_PATH} && echo 'exists' || echo 'missing'" 2>/dev/null | grep -q "exists"; then
    echo "✓ Repository found at ${REMOTE_PATH}"
    
    # Check repo structure
    echo ""
    echo "[4/7] Checking repository structure..."
    REPO_STRUCTURE=$(${SSH_CMD} root@${POD_IP} "cd ${REMOTE_PATH} && ls -la 2>/dev/null | head -20" 2>/dev/null)
    echo "  Repository contents:"
    echo "$REPO_STRUCTURE" | sed 's/^/    /'
    
    # Check key directories
    echo ""
    echo "[5/7] Checking key directories..."
    for dir in "notebooks" "data/processed" "data/labeled" "figures" "logs"; do
        if ${SSH_CMD} root@${POD_IP} "test -d ${REMOTE_PATH}/${dir} && echo 'exists' || echo 'missing'" 2>/dev/null | grep -q "exists"; then
            FILE_COUNT=$(${SSH_CMD} root@${POD_IP} "find ${REMOTE_PATH}/${dir} -type f 2>/dev/null | wc -l" 2>/dev/null)
            echo "  ✓ ${dir}/ ($FILE_COUNT files)"
        else
            echo "  ✗ ${dir}/ (missing)"
        fi
    done
    
    # Check for specific file types
    echo ""
    echo "[6/7] Checking for data files..."
    for pattern in "*.parquet" "*.pkl" "*.npy" "*.npz" "*.ipynb"; do
        FILE_COUNT=$(${SSH_CMD} root@${POD_IP} "find ${REMOTE_PATH} -name '${pattern}' -type f 2>/dev/null | wc -l" 2>/dev/null)
        if [ "$FILE_COUNT" -gt 0 ]; then
            echo "  ✓ ${pattern}: $FILE_COUNT files found"
            # Show first few
            FIRST_FILES=$(${SSH_CMD} root@${POD_IP} "find ${REMOTE_PATH} -name '${pattern}' -type f 2>/dev/null | head -3" 2>/dev/null)
            echo "$FIRST_FILES" | sed 's|^.*action-grounding/|    - |'
        else
            echo "  ✗ ${pattern}: none found"
        fi
    done
    
    # Check git status
    echo ""
    echo "[7/7] Checking git status..."
    GIT_STATUS=$(${SSH_CMD} root@${POD_IP} "cd ${REMOTE_PATH} && git status --short 2>/dev/null | head -5" 2>/dev/null)
    if [ -n "$GIT_STATUS" ]; then
        echo "  Git changes:"
        echo "$GIT_STATUS" | sed 's/^/    /'
    else
        echo "  ✓ No uncommitted changes"
    fi
    
else
    echo "❌ Repository not found at ${REMOTE_PATH}"
    echo ""
    echo "The repository hasn't been cloned yet. You need to:"
    echo ""
    echo "1. SSH into the pod:"
    echo "   ssh root@${POD_IP} -p ${PORT}"
    echo ""
    echo "2. Clone the repository:"
    echo "   cd /workspace"
    echo "   git clone https://github.com/YOUR_USERNAME/action-grounding.git"
    echo "   cd action-grounding"
    echo ""
    echo "3. Run the setup script:"
    echo "   bash <(curl -s https://raw.githubusercontent.com/YOUR_USERNAME/action-grounding/main/scripts/setup_runpod.sh)"
    echo "   # Or manually run: ./scripts/setup_runpod.sh"
fi

echo ""
echo "=================================================="
echo "Status check complete!"
echo "=================================================="

