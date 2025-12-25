#!/bin/bash
# Save RunPod connection details for easy reuse
# Usage: ./scripts/save_pod_connection.sh

CONFIG_FILE="$HOME/.runpod_connection"

echo "=================================================="
echo "RunPod Connection Details Setup"
echo "=================================================="
echo ""
echo "Enter your RunPod connection details:"
echo ""

read -p "Pod IP address (e.g., 123.45.67.89): " POD_IP
read -p "SSH Port (e.g., 12345): " SSH_PORT
read -p "Pod ID (optional, for reference): " POD_ID

# Validate inputs
if [ -z "$POD_IP" ] || [ -z "$SSH_PORT" ]; then
    echo "❌ Error: Pod IP and SSH Port are required"
    exit 1
fi

# Save to config file
cat > "$CONFIG_FILE" << EOF
# RunPod Connection Details
# Saved on $(date)
POD_IP=$POD_IP
SSH_PORT=$SSH_PORT
POD_ID=$POD_ID
EOF

echo ""
echo "✓ Connection details saved to: $CONFIG_FILE"
echo ""
echo "Your connection details:"
echo "  Pod IP: $POD_IP"
echo "  SSH Port: $SSH_PORT"
[ -n "$POD_ID" ] && echo "  Pod ID: $POD_ID"
echo ""
echo "You can now use these commands:"
echo ""
echo "  # SSH into pod:"
echo "  ssh root@$POD_IP -p $SSH_PORT"
echo ""
echo "  # Sync from pod:"
echo "  ./scripts/sync_from_pod.sh $POD_IP $SSH_PORT"
echo ""
echo "To update these details, run this script again."

