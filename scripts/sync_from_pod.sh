#!/bin/bash
# Sync everything from RunPod pod to local Mac
# Usage: ./scripts/sync_from_pod.sh [POD_IP] [PORT]
#        If POD_IP/PORT not provided, tries to load from ~/.runpod_connection

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
    echo ""
    if [ -f "$CONFIG_FILE" ]; then
        echo "Note: Config file exists at $CONFIG_FILE but missing POD_IP or SSH_PORT"
    else
        echo "Tip: Run './scripts/save_pod_connection.sh' to save connection details"
    fi
    exit 1
fi

REMOTE_PATH="/workspace/action-grounding"
LOCAL_PATH="/Users/gadi/Personal/interpret"

echo "=================================================="
echo "Syncing from RunPod pod..."
echo "Pod: root@${POD_IP}:${PORT}"
echo "=================================================="

# Check if rsync is available on remote pod
echo ""
echo "Checking for rsync on remote pod..."
if ssh -p ${PORT} root@${POD_IP} "which rsync > /dev/null 2>&1"; then
    USE_RSYNC=true
    echo "✓ rsync available on remote pod"
else
    USE_RSYNC=false
    echo "⚠ rsync not found on remote pod, using scp instead"
    echo "  (To install rsync on pod: apt-get update && apt-get install -y rsync)"
fi

# Create local directories if they don't exist
mkdir -p "${LOCAL_PATH}/figures"
mkdir -p "${LOCAL_PATH}/data/processed"
mkdir -p "${LOCAL_PATH}/data/labeled"
mkdir -p "${LOCAL_PATH}/notebooks"
mkdir -p "${LOCAL_PATH}/logs"

# Sync figures (usually small, fast)
echo ""
echo "[1/6] Syncing figures..."
if ssh -p ${PORT} root@${POD_IP} "test -d ${REMOTE_PATH}/figures" 2>/dev/null; then
    if [ "$USE_RSYNC" = true ]; then
        rsync -avz --progress -e "ssh -p ${PORT}" \
          root@${POD_IP}:${REMOTE_PATH}/figures/ \
          "${LOCAL_PATH}/figures/"
    else
        scp -P ${PORT} -r root@${POD_IP}:${REMOTE_PATH}/figures/* "${LOCAL_PATH}/figures/" 2>/dev/null || true
    fi
    echo "  ✓ Figures synced"
else
    echo "  (No figures directory)"
fi

# Sync processed data (parquet, pkl - moderate size)
echo ""
echo "[2/6] Syncing processed data (parquet, pkl)..."
if ssh -p ${PORT} root@${POD_IP} "test -d ${REMOTE_PATH}/data/processed" 2>/dev/null; then
    # Sync parquet files
    PARQUET_FILES=$(ssh -p ${PORT} root@${POD_IP} "find ${REMOTE_PATH}/data/processed -maxdepth 1 -name '*.parquet' -type f 2>/dev/null" | wc -l)
    if [ "$PARQUET_FILES" -gt 0 ]; then
        if [ "$USE_RSYNC" = true ]; then
            rsync -avz --progress -e "ssh -p ${PORT}" \
              root@${POD_IP}:${REMOTE_PATH}/data/processed/*.parquet \
              "${LOCAL_PATH}/data/processed/" 2>/dev/null
        else
            ssh -p ${PORT} root@${POD_IP} "cd ${REMOTE_PATH}/data/processed && ls *.parquet 2>/dev/null" | while read file; do
                scp -P ${PORT} root@${POD_IP}:${REMOTE_PATH}/data/processed/"$file" "${LOCAL_PATH}/data/processed/" 2>/dev/null
            done
        fi
        echo "  ✓ Parquet files synced ($PARQUET_FILES files)"
    else
        echo "  (No parquet files)"
    fi
    
    # Sync pkl files
    PKL_FILES=$(ssh -p ${PORT} root@${POD_IP} "find ${REMOTE_PATH}/data/processed -maxdepth 1 -name '*.pkl' -type f 2>/dev/null" | wc -l)
    if [ "$PKL_FILES" -gt 0 ]; then
        if [ "$USE_RSYNC" = true ]; then
            rsync -avz --progress -e "ssh -p ${PORT}" \
              root@${POD_IP}:${REMOTE_PATH}/data/processed/*.pkl \
              "${LOCAL_PATH}/data/processed/" 2>/dev/null
        else
            ssh -p ${PORT} root@${POD_IP} "cd ${REMOTE_PATH}/data/processed && ls *.pkl 2>/dev/null" | while read file; do
                scp -P ${PORT} root@${POD_IP}:${REMOTE_PATH}/data/processed/"$file" "${LOCAL_PATH}/data/processed/" 2>/dev/null
            done
        fi
        echo "  ✓ PKL files synced ($PKL_FILES files)"
    else
        echo "  (No pkl files)"
    fi
else
    echo "  (No data/processed directory)"
fi

# Sync notebooks (to get any updated results/cells)
echo ""
echo "[3/6] Syncing notebooks..."
if ssh -p ${PORT} root@${POD_IP} "test -d ${REMOTE_PATH}/notebooks" 2>/dev/null; then
    NOTEBOOK_FILES=$(ssh -p ${PORT} root@${POD_IP} "find ${REMOTE_PATH}/notebooks -maxdepth 1 -name '*.ipynb' -type f 2>/dev/null" | wc -l)
    if [ "$NOTEBOOK_FILES" -gt 0 ]; then
        if [ "$USE_RSYNC" = true ]; then
            rsync -avz --progress -e "ssh -p ${PORT}" \
              root@${POD_IP}:${REMOTE_PATH}/notebooks/*.ipynb \
              "${LOCAL_PATH}/notebooks/" 2>/dev/null
        else
            ssh -p ${PORT} root@${POD_IP} "cd ${REMOTE_PATH}/notebooks && ls *.ipynb 2>/dev/null" | while read file; do
                scp -P ${PORT} root@${POD_IP}:${REMOTE_PATH}/notebooks/"$file" "${LOCAL_PATH}/notebooks/" 2>/dev/null
            done
        fi
        echo "  ✓ Notebooks synced ($NOTEBOOK_FILES files)"
    else
        echo "  (No notebook files)"
    fi
else
    echo "  (No notebooks directory)"
fi

# Sync logs (optional)
echo ""
echo "[4/6] Syncing logs..."
if ssh -p ${PORT} root@${POD_IP} "test -d ${REMOTE_PATH}/logs" 2>/dev/null; then
    if [ "$USE_RSYNC" = true ]; then
        rsync -avz --progress -e "ssh -p ${PORT}" \
          root@${POD_IP}:${REMOTE_PATH}/logs/ \
          "${LOCAL_PATH}/logs/"
    else
        scp -P ${PORT} -r root@${POD_IP}:${REMOTE_PATH}/logs/* "${LOCAL_PATH}/logs/" 2>/dev/null || true
    fi
    echo "  ✓ Logs synced"
else
    echo "  (No logs directory)"
fi

# Ask about large files
echo ""
echo "[5/6] Large activation files (.npy, .npz)..."
read -p "Sync large activation files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Sync .npy files from processed
    if ssh -p ${PORT} root@${POD_IP} "test -d ${REMOTE_PATH}/data/processed" 2>/dev/null; then
        NPY_FILES=$(ssh -p ${PORT} root@${POD_IP} "find ${REMOTE_PATH}/data/processed -maxdepth 1 -name '*.npy' -type f 2>/dev/null" | wc -l)
        if [ "$NPY_FILES" -gt 0 ]; then
            echo "  Syncing .npy files ($NPY_FILES files, this may take a while)..."
            if [ "$USE_RSYNC" = true ]; then
                rsync -avz --progress -e "ssh -p ${PORT}" \
                  root@${POD_IP}:${REMOTE_PATH}/data/processed/*.npy \
                  "${LOCAL_PATH}/data/processed/" 2>/dev/null
            else
                ssh -p ${PORT} root@${POD_IP} "cd ${REMOTE_PATH}/data/processed && ls *.npy 2>/dev/null" | while read file; do
                    scp -P ${PORT} root@${POD_IP}:${REMOTE_PATH}/data/processed/"$file" "${LOCAL_PATH}/data/processed/" 2>/dev/null
                done
            fi
            echo "    ✓ .npy files synced"
        else
            echo "    (No .npy files in data/processed)"
        fi
        
        # Sync .npz files from processed
        NPZ_FILES=$(ssh -p ${PORT} root@${POD_IP} "find ${REMOTE_PATH}/data/processed -maxdepth 1 -name '*.npz' -type f 2>/dev/null" | wc -l)
        if [ "$NPZ_FILES" -gt 0 ]; then
            echo "  Syncing .npz files ($NPZ_FILES files)..."
            if [ "$USE_RSYNC" = true ]; then
                rsync -avz --progress -e "ssh -p ${PORT}" \
                  root@${POD_IP}:${REMOTE_PATH}/data/processed/*.npz \
                  "${LOCAL_PATH}/data/processed/" 2>/dev/null
            else
                ssh -p ${PORT} root@${POD_IP} "cd ${REMOTE_PATH}/data/processed && ls *.npz 2>/dev/null" | while read file; do
                    scp -P ${PORT} root@${POD_IP}:${REMOTE_PATH}/data/processed/"$file" "${LOCAL_PATH}/data/processed/" 2>/dev/null
                done
            fi
            echo "    ✓ .npz files synced"
        else
            echo "    (No .npz files in data/processed)"
        fi
    fi
    
    # Sync labeled .npz files
    if ssh -p ${PORT} root@${POD_IP} "test -d ${REMOTE_PATH}/data/labeled" 2>/dev/null; then
        LABELED_NPZ=$(ssh -p ${PORT} root@${POD_IP} "find ${REMOTE_PATH}/data/labeled -maxdepth 1 -name '*.npz' -type f 2>/dev/null" | wc -l)
        if [ "$LABELED_NPZ" -gt 0 ]; then
            echo "  Syncing labeled .npz files ($LABELED_NPZ files)..."
            if [ "$USE_RSYNC" = true ]; then
                rsync -avz --progress -e "ssh -p ${PORT}" \
                  root@${POD_IP}:${REMOTE_PATH}/data/labeled/*.npz \
                  "${LOCAL_PATH}/data/labeled/" 2>/dev/null
            else
                ssh -p ${PORT} root@${POD_IP} "cd ${REMOTE_PATH}/data/labeled && ls *.npz 2>/dev/null" | while read file; do
                    scp -P ${PORT} root@${POD_IP}:${REMOTE_PATH}/data/labeled/"$file" "${LOCAL_PATH}/data/labeled/" 2>/dev/null
                done
            fi
            echo "    ✓ Labeled .npz files synced"
        else
            echo "    (No .npz files in data/labeled)"
        fi
    fi
else
    echo "  Skipping large files (use manual sync if needed)"
fi

# Sync any other important files
echo ""
echo "[6/6] Syncing other files..."
if ssh -p ${PORT} root@${POD_IP} "test -d ${REMOTE_PATH}/data/processed" 2>/dev/null; then
    JSON_FILES=$(ssh -p ${PORT} root@${POD_IP} "find ${REMOTE_PATH}/data/processed -maxdepth 1 -name '*.json' -type f 2>/dev/null" | wc -l)
    if [ "$JSON_FILES" -gt 0 ]; then
        if [ "$USE_RSYNC" = true ]; then
            rsync -avz --progress -e "ssh -p ${PORT}" \
              root@${POD_IP}:${REMOTE_PATH}/data/processed/*.json \
              "${LOCAL_PATH}/data/processed/" 2>/dev/null
        else
            ssh -p ${PORT} root@${POD_IP} "cd ${REMOTE_PATH}/data/processed && ls *.json 2>/dev/null" | while read file; do
                scp -P ${PORT} root@${POD_IP}:${REMOTE_PATH}/data/processed/"$file" "${LOCAL_PATH}/data/processed/" 2>/dev/null
            done
        fi
        echo "  ✓ JSON files synced ($JSON_FILES files)"
    else
        echo "  (No json files)"
    fi
else
    echo "  (No data/processed directory)"
fi

echo ""
echo "=================================================="
echo "✓ Sync complete!"
echo "=================================================="

# Check if anything was actually synced
SYNCED_ANYTHING=false
if [ -d "${LOCAL_PATH}/figures" ] && [ "$(ls -A ${LOCAL_PATH}/figures 2>/dev/null)" ]; then
    SYNCED_ANYTHING=true
elif [ -d "${LOCAL_PATH}/data/processed" ] && [ "$(ls -A ${LOCAL_PATH}/data/processed 2>/dev/null)" ]; then
    SYNCED_ANYTHING=true
elif [ -f "${LOCAL_PATH}/notebooks"/*.ipynb ] 2>/dev/null; then
    SYNCED_ANYTHING=true
fi

if [ "$SYNCED_ANYTHING" = false ]; then
    echo ""
    echo "⚠ No files were synced. This could mean:"
    echo "  1. The repository hasn't been cloned on the pod yet"
    echo "  2. No work has been done on the pod yet"
    echo "  3. Files are in a different location"
    echo ""
    echo "Check pod status:"
    echo "  ./scripts/check_pod_status.sh ${POD_IP} ${PORT}"
    echo ""
    echo "Or SSH into the pod to investigate:"
    echo "  ssh root@${POD_IP} -p ${PORT}"
    echo "  cd /workspace/action-grounding"
    echo "  ls -la"
fi

echo ""
echo "Synced to: ${LOCAL_PATH}"
echo ""
echo "To sync again, run:"
echo "  ./scripts/sync_from_pod.sh ${POD_IP} ${PORT}"

