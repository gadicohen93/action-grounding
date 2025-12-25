#!/bin/bash
# RunPod Setup Script
# Run this on your RunPod pod after first SSH

set -e  # Exit on error

echo "=================================================="
echo "RunPod Setup Script for Action Grounding Research"
echo "=================================================="

# Check if running on GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. Are you on a GPU pod?"
    exit 1
fi

echo "✓ GPU detected"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Update system packages
echo ""
echo "[1/8] Updating system packages..."
apt-get update -qq
apt-get install -y git tmux htop vim curl wget rsync > /dev/null 2>&1
echo "✓ System packages installed (including rsync for syncing)"

# Navigate to workspace
echo ""
echo "[2/8] Setting up workspace..."
cd /workspace || cd /root
mkdir -p action-grounding
cd action-grounding
echo "✓ Workspace ready: $(pwd)"

# Check if repo exists
if [ -d ".git" ]; then
    echo ""
    echo "[3/8] Git repo found, pulling latest..."
    git pull
else
    echo ""
    echo "[3/8] Clone your repository manually:"
    echo "    git clone https://github.com/YOUR_USERNAME/action-grounding.git"
    echo "    cd action-grounding"
    echo "Then run this script again."
    exit 0
fi

# Check for SSH key (for GitHub push access)
echo ""
if [ ! -f ~/.ssh/id_ed25519.pub ] && [ ! -f ~/.ssh/id_rsa.pub ]; then
    echo "[3.5/9] No SSH key found. To enable git push, generate one:"
    echo "    ssh-keygen -t ed25519 -C 'your_email@example.com'"
    echo "    cat ~/.ssh/id_ed25519.pub"
    echo "    (Then add to GitHub: Settings → SSH and GPG keys)"
    echo "    git remote set-url origin git@github.com:YOUR_USERNAME/action-grounding.git"
else
    echo "[3.5/9] SSH key found:"
    ls -1 ~/.ssh/*.pub 2>/dev/null | head -1 | xargs cat
    echo ""
    echo "    (If not added to GitHub, add the key above to GitHub Settings → SSH and GPG keys)"
fi

# Create Python virtual environment
echo ""
echo "[4/8] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✓ Virtual environment created"

# Upgrade pip
echo ""
echo "[5/8] Upgrading pip..."
pip install --upgrade pip -q
echo "✓ pip upgraded"

# Install dependencies
echo ""
echo "[6/8] Installing Python dependencies..."
echo "    (This takes 5-10 minutes)"
pip install -r requirements.txt -q
echo "✓ Dependencies installed"

# Verify PyTorch GPU
echo ""
echo "[7/9] Verifying PyTorch GPU support..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# Register Jupyter kernel
echo ""
echo "[8/10] Registering Jupyter kernel..."
python -m ipykernel install --user --name=interpret --display-name="Python (interpret venv)"
echo "✓ Jupyter kernel registered"

# Set up .env file
echo ""
echo "[9/10] Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚠ .env file created from template"
    echo "   ACTION REQUIRED: Edit .env with your API keys:"
    echo "   nano .env"
else
    echo "✓ .env file already exists"
fi

# Create data directories
mkdir -p data/raw data/processed figures logs

# Final step: Git remote check
echo ""
echo "[10/10] Checking Git remote configuration..."
if git remote get-url origin 2>/dev/null | grep -q "^https://"; then
    echo "⚠ Git remote uses HTTPS. To enable push without password prompts:"
    echo "   Option 1: Switch to SSH (recommended)"
    echo "      git remote set-url origin git@github.com:YOUR_USERNAME/action-grounding.git"
    echo "      (Make sure SSH key is added to GitHub)"
    echo "   Option 2: Use Personal Access Token when pushing"
    echo "      (Generate at: GitHub → Settings → Developer settings → Personal access tokens)"
fi

echo ""
echo "=================================================="
echo "✓ Setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys:"
echo "   nano .env"
echo ""
echo "2. Verify configuration:"
echo "   python -c 'from src.config import get_config, get_secrets; get_config(); print(\"Config OK\")'"
echo ""
echo "3. Start Jupyter Lab:"
echo "   tmux new -s jupyter"
echo "   source venv/bin/activate"
echo "   jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root"
echo ""
echo "4. Access Jupyter via RunPod Connect → Jupyter button"
echo ""
echo "5. Run notebooks in order: 01 → 02 → 03 → 04"
echo ""
echo "=================================================="
