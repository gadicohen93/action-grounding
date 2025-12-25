# RunPod Deployment Guide: Step-by-Step

## Quick Reference: Finding Connection Details

**Where to find your Pod IP and Port:**

1. Go to: https://runpod.io/console/pods
2. Click on your running pod
3. Look for **"SSH"** section - you'll see: `ssh root@X.X.X.X -p XXXXX`
   - **POD_IP** = the IP address (X.X.X.X)
   - **SSH_PORT** = the port number (XXXXX)
4. Save these for easy reuse: `./scripts/save_pod_connection.sh`

**Common Ports:**
- **SSH Port**: Random 5-digit number assigned by RunPod (e.g., 12345, 23456)
- **Jupyter Port**: Internal port 8888 (mapped to external port by RunPod)
- **Pod IP**: Changes if you stop/restart pod

See **Part 2.3** for detailed instructions with screenshots.

---

## Prerequisites

- RunPod account (runpod.io)
- Git repository with your code (GitHub/GitLab)
- API keys: HuggingFace token, OpenAI API key

---

## Part 1: Local Preparation (5 minutes)

### 1.1 Commit and Push Your Code

```bash
# On your Mac
cd /Users/gadi/Personal/interpret

# Remove API keys from any files (they're in .env now)
git status

# Commit refactored code
git add .
git commit -m "Refactored for GPU deployment"

# Push to remote (create repo if needed)
# If no remote:
gh repo create action-grounding --private --source=. --remote=origin --push
# Or use GitHub web UI to create repo, then:
git remote add origin https://github.com/YOUR_USERNAME/action-grounding.git
git push -u origin main
```

### 1.2 Verify Critical Files

Ensure these exist:
- [ ] `config.yaml`
- [ ] `.env.example`
- [ ] `requirements.txt`
- [ ] `src/` directory with all modules
- [ ] `notebooks/` with 4 notebooks

---

## Part 2: Create RunPod GPU Pod (10 minutes)

### 2.1 Choose GPU

Go to **runpod.io/console/gpu-cloud**

**Recommended GPUs** (by budget):

| GPU | VRAM | Speed | Cost/hr | Recommendation |
|-----|------|-------|---------|----------------|
| **RTX 4090** | 24GB | Fast | ~$0.40 | Best value |
| **RTX 3090** | 24GB | Good | ~$0.30 | Budget option |
| **A100 40GB** | 40GB | Fastest | ~$1.00 | If you need full precision |
| **RTX A5000** | 24GB | Good | ~$0.50 | Stable option |

For this project: **RTX 4090 or 3090 is perfect** (Mistral-7B in 8-bit needs ~8GB)

### 2.2 Deploy Pod

1. Click **"Deploy"** on your chosen GPU
2. **Template:** Select **"RunPod PyTorch 2.1"** or **"PyTorch"**
3. **Container Disk:** 50 GB (minimum)
4. **Volume (Persistent Storage):**
   - If available, add 100 GB volume (keeps data between stops)
   - Mount point: `/workspace`
5. **Expose Ports:**
   - HTTP Port: 8888 (Jupyter)
   - SSH: Enable
6. Click **"Deploy On-Demand"**

**Wait 1-2 minutes** for pod to start.

### 2.3 Get Connection Info

Once your pod is running, you need to find these connection details:

#### Step-by-Step: Finding Your Connection Details

1. **Go to RunPod Console**
   - Navigate to: https://runpod.io/console/pods
   - You should see your pod listed (status: "Running")

2. **Click on Your Pod**
   - Click the pod name/row to open pod details

3. **Find Connection Information**
   
   You'll see several tabs/sections. Look for:

   **A. SSH Connection Details**
   - Look for a section labeled **"SSH"** or **"Connect"**
   - You should see something like:
     ```
     ssh root@123.45.67.89 -p 12345
     ```
   - **POD_IP** = `123.45.67.89` (the IP address)
   - **SSH_PORT** = `12345` (the port number after `-p`)
   
   **B. Jupyter Connection**
   - Look for **"Connect"** button or **"Jupyter"** section
   - May show a URL like: `https://12345-abc.runpod.net` or similar
   - Or click **"Connect"** → **"Start Jupyter Lab"** button
   
   **C. Pod Details**
   - **Pod ID**: Usually shown at the top (e.g., `abc123def456`)
   - **Pod Name**: Your custom name or auto-generated name

#### Visual Guide: What to Look For

In the RunPod UI, you'll typically see:

```
┌─────────────────────────────────────────┐
│ Pod: my-pod-name                       │
│ Status: Running                        │
│                                         │
│ SSH:                                    │
│   ssh root@123.45.67.89 -p 12345       │
│                                         │
│ Connect:                                │
│   [Start Jupyter Lab] [Connect]        │
│                                         │
│ Pod ID: abc123def456                   │
└─────────────────────────────────────────┘
```

#### Save These Values

**Option A: Use the Helper Script (Recommended)**

Save your connection details for easy reuse:

```bash
# On your Mac
cd /Users/gadi/Personal/interpret
./scripts/save_pod_connection.sh

# Follow the prompts to enter:
# - Pod IP address
# - SSH Port
# - Pod ID (optional)
```

This saves to `~/.runpod_connection` and you can use scripts without typing IP/port each time.

**Option B: Manual Note**

Create a note with:
- **POD_IP**: `123.45.67.89` (example - use your actual IP)
- **SSH_PORT**: `12345` (example - use your actual port)
- **Pod ID**: `abc123def456` (for reference)

**Quick Test:**
```bash
# From your Mac terminal, test SSH connection:
ssh root@<POD_IP> -p <SSH_PORT>

# Example:
ssh root@123.45.67.89 -p 12345
```

If it works, you'll see the pod's terminal prompt.

#### Alternative: Using RunPod CLI

If you have RunPod CLI installed:
```bash
# List all pods
runpod pod list

# Get pod details (shows connection info)
runpod pod get <POD_ID>
```

#### Common Issues

**Can't find SSH details?**
- Make sure SSH was enabled during pod creation (step 2.2)
- Check if pod is fully started (wait 1-2 minutes)
- Look in "Settings" or "Network" tab of pod details

**Port number seems wrong?**
- RunPod assigns random ports for security
- The port is usually 5 digits (e.g., 12345, 23456)
- It's different from the internal port 8888 (Jupyter uses 8888 internally, but RunPod maps it to a different external port)

**Need to reconnect later?**
- Pod IP and port stay the same while pod is running
- If you stop/restart pod, ports may change
- Always check pod details after restarting

---

## Part 3: Initial Setup on Pod (15 minutes)

### 3.1 SSH Into Pod

```bash
# From your Mac terminal
ssh root@<POD_IP> -p <PORT>

# First time: confirm fingerprint (type "yes")
```

You should see a prompt like: `root@<pod-id>:~#`

### 3.2 Install Essential Tools

```bash
# Update package manager
apt-get update

# Install git and other essentials
apt-get install -y git tmux htop vim

# Verify GPU
nvidia-smi
# Should show your GPU (RTX 4090, etc.)

# Verify PyTorch sees GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: CUDA available: True
```

### 3.3 Clone Your Repository

```bash
# Navigate to persistent storage
cd /workspace

# Clone your repo
git clone https://github.com/YOUR_USERNAME/action-grounding.git
cd action-grounding

# Verify files
ls -la
# Should see: src/, notebooks/, config.yaml, requirements.txt, etc.
```

### 3.3a Set Up GitHub SSH Access (For Git Push)

If you want to push changes from the pod to GitHub, you need to set up SSH keys:

**Option A: Generate New SSH Key on Pod (Recommended)**

```bash
# On the pod, generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter to accept default location (~/.ssh/id_ed25519)
# Press Enter twice for no passphrase (or set one if preferred)

# Display the public key
cat ~/.ssh/id_ed25519.pub
# Copy the entire output (starts with ssh-ed25519...)
```

**Add Key to GitHub:**
1. Go to GitHub.com → Settings → SSH and GPG keys
2. Click "New SSH key"
3. Paste the public key you copied
4. Give it a title like "RunPod Pod"
5. Click "Add SSH key"

**Configure Git to Use SSH:**
```bash
# If you cloned with HTTPS, switch to SSH
cd /workspace/action-grounding
git remote set-url origin git@github.com:YOUR_USERNAME/action-grounding.git

# Test SSH connection
ssh -T git@github.com
# Should see: "Hi YOUR_USERNAME! You've successfully authenticated..."

# Now you can push
git push
```

**Option B: Use HTTPS with Personal Access Token**

If you prefer HTTPS (no SSH setup needed):

```bash
# Clone/configure with HTTPS
git remote set-url origin https://github.com/YOUR_USERNAME/action-grounding.git

# When pushing, use a Personal Access Token as password:
# 1. Go to GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
# 2. Generate new token with 'repo' scope
# 3. When git asks for password, paste the token (not your GitHub password)
git push
# Username: YOUR_USERNAME
# Password: <paste token here>
```

**Note:** SSH keys persist across pod restarts if you're using a persistent volume at `/workspace` or `/root`. If the pod is ephemeral, you'll need to regenerate keys each time.

### 3.4 Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
# This takes 5-10 minutes
pip install -r requirements.txt

# Verify key packages
python -c "import torch; import transformers; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Register venv as Jupyter kernel
python -m ipykernel install --user --name=interpret --display-name="Python (interpret venv)"
```

**Important:** After registering the kernel, you'll need to select it in your notebooks:
- In Jupyter Notebook: **Kernel** → **Change Kernel** → **"Python (interpret venv)"**
- In JupyterLab: Click kernel name (top-right) → Select **"Python (interpret venv)"**

### 3.5 Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit with your keys
nano .env
# Or use vim: vim .env
```

Add your actual keys:
```bash
HF_TOKEN=hf_your_actual_token_here
OPENAI_API_KEY=sk-your_actual_key_here
```

Save and exit (`Ctrl+X`, then `Y`, then `Enter` in nano)

### 3.6 Verify Configuration

```bash
# Check config loads
python -c "from src.config import get_config, get_secrets; c = get_config(); s = get_secrets(); print('Config OK'); print('HF Token:', 'SET' if s.hf_token else 'MISSING'); print('OpenAI Key:', 'SET' if s.openai_api_key else 'MISSING')"
```

Should print:
```
Config OK
HF Token: SET
OpenAI Key: SET
```

---

## Part 4: Launch Jupyter (5 minutes)

### 4.1 Start Jupyter in tmux (Recommended)

Using `tmux` lets you detach and keep Jupyter running:

```bash
# Start tmux session
tmux new -s jupyter

# Activate venv (if not already)
source venv/bin/activate

# Start Jupyter Lab
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root

# You'll see output like:
#   http://127.0.0.1:8888/lab?token=abc123...
```

**Don't close this!** Leave it running.

### 4.2 Access Jupyter in Browser

**Option A: RunPod Web UI**
1. Go back to RunPod console
2. Click your pod
3. Click **"Connect"** → **"Start Jupyter Lab"** or the Jupyter button
4. Should open Jupyter in browser automatically

**Option B: Direct URL**
1. In RunPod pod details, find **"Jupyter URL"**
2. Copy and paste into browser
3. If prompted for token, get it from the terminal output

### 4.3 Detach from tmux (Optional)

Press: `Ctrl+B`, then `D`

This detaches but keeps Jupyter running. To reattach later:
```bash
tmux attach -t jupyter
```

---

## Part 5: Run Notebooks (6-9 hours GPU time)

### 5.1 Open Jupyter

In Jupyter Lab:
1. Navigate to `notebooks/` folder
2. You should see: `01_behavioral_phenomenon.ipynb`, etc.
3. **IMPORTANT:** When opening a notebook, select the correct kernel:
   - Click the kernel name in the top-right (e.g., "Python 3")
   - Select **"Python (interpret venv)"** from the dropdown
   - This ensures notebooks use your venv with all installed packages

### 5.2 Notebook 01: Behavioral Phenomenon (~2-4 hours)

1. Open `01_behavioral_phenomenon.ipynb`
2. **Before running:** Check config in the notebook
3. Run cells sequentially (Shift+Enter)

**Critical cell:**
```python
episodes = generate_batch(
    conditions=conditions,
    n_per_condition=config.experiment.n_episodes_per_condition,  # 50
    model_id=config.model.id,
    labeling_method="openai",
    save_path=config.data.processed_dir / "episodes.parquet",
    verbose=True,
)
```

**Expected:**
- Progress bar showing episode generation
- OpenAI API calls for labeling
- Takes 2-4 hours for 2,250 episodes
- Outputs: `data/processed/episodes.parquet`

**Check results:**
```python
# Last cell should show:
# Fake rate: 25-30%
# Chi-squared: p < 0.001
# Figure saved to figures/figure1_fake_rates.pdf
```

### 5.3 Notebook 02: Mechanistic Probes (~1-2 hours)

**Prerequisites:** `episodes.parquet` from Notebook 01

1. Open `02_mechanistic_probes.ipynb`
2. Run cells sequentially

**Critical cells:**

```python
# Activation extraction (takes 1-2 hours)
dataset = extract_activations_batch(
    episodes=episodes,
    positions=config.extraction.positions,
    layers=config.extraction.layers,
    model_id=config.model.id,
    save_path=config.data.processed_dir / "activations.parquet",
    verbose=True,
)
```

**CRITICAL RESULT TO CHECK:**
```python
# Position analysis cell
position_accuracies = {...}
first_assistant_acc = position_accuracies.get('first_assistant', 0)

# This MUST be > 80% for strong application
if first_assistant_acc > 0.80:
    print("✓ Probe detects action-grounding, not syntax!")
```

**Expected outputs:**
- `data/processed/activations.parquet`
- `data/processed/reality_probe.pkl`
- `figures/figure2_position_accuracy.pdf` ← **CRITICAL FIGURE**
- `figures/figure3_fake_vs_true_probs.png`

### 5.4 Notebook 03: Generalization (~30 minutes)

**Prerequisites:** `activations.parquet` from Notebook 02

1. Open `03_generalization.ipynb`
2. Run cells sequentially
3. Much faster (no new generation/extraction)

**Critical result:**
```python
mean_cross_tool = ...
# Should be > 85% for strong generalization claim
```

**Expected outputs:**
- `figures/figure4_transfer_matrix.pdf`
- `figures/figure5_layer_accuracy.pdf`
- `figures/tsne_by_tool.png`

### 5.5 Notebook 04: Causal Intervention (~2 hours)

**Prerequisites:** `episodes.parquet` + `reality_probe.pkl`

1. Open `04_causal_intervention.ipynb`
2. Run cells sequentially
3. **This is the longest** (steering requires many forward passes)

**Critical result:**
```python
fake_effect_size = ...
# > 20% = strong causal evidence
# 10-20% = moderate evidence
# < 10% = weak/null (still reportable!)
```

**Expected outputs:**
- `figures/figure6_steering_dose_response.pdf`
- Steering results logged

---

## Part 6: Monitoring & Troubleshooting

### 6.1 Monitor GPU Usage

In a **second SSH session** (or new tmux pane):

```bash
# SSH into pod again
ssh root@<POD_IP> -p <PORT>

# Watch GPU usage
watch -n 1 nvidia-smi
```

Should show:
- GPU utilization: 80-100% when running
- Memory usage: ~8-12 GB for 8-bit Mistral-7B
- Temperature: Monitor if it stays reasonable

### 6.2 Monitor Logs

If you set up logging to file:

```bash
tail -f logs/experiment.log
```

Or in Jupyter, check cell outputs for progress bars.

### 6.3 Common Issues

**Issue: Out of Memory**
```
RuntimeError: CUDA out of memory
```

**Fix:**
```bash
# Edit config.yaml
nano config.yaml
# Change:
# model.quantization: "8bit"  # or "4bit"
# extraction.batch_size: 4  # reduce from 8
```

**Issue: OpenAI rate limit**
```
RateLimitError: You exceeded your quota
```

**Fix:**
- Wait a bit and retry
- Or switch to regex labeling (less accurate):
  ```python
  labeling_method="regex"  # in generate_batch call
  ```

**Issue: Model download slow**
```
Downloading model... (hanging)
```

**Fix:**
- RunPod has fast internet, but Mistral-7B is ~14GB
- First download takes 5-10 minutes
- Check: `ls ~/.cache/huggingface/hub/` to see progress

**Issue: Git push fails - file too large**
```
remote: error: File notebooks/data/processed/activations.npy is 210.94 MB; 
this exceeds GitHub's file size limit of 100.00 MB
```

**Fix:** Remove large data files from git (they shouldn't be version controlled):

```bash
# On the pod (or locally)
cd /workspace/action-grounding

# Remove the large file from git tracking (but keep it locally)
git rm --cached notebooks/data/processed/activations.npy
# Or if it's in data/processed/:
git rm --cached data/processed/*.npy data/processed/*.npz

# Make sure .gitignore has these patterns:
# *.npy
# *.npz
# data/**/*.parquet
# notebooks/data/**/*.npy

# Commit the removal
git add .gitignore
git commit -m "Remove large data files from git tracking"

# Now push should work
git push
```

**Note:** Large data files (activations, processed datasets) should be:
- Stored locally on the pod (in `/workspace` persistent volume)
- Synced back to your Mac via `scp` or `rsync` (see Part 9)
- **NOT** committed to git (use `.gitignore`)

If you really need version control for large files, use Git LFS:
```bash
# Install git-lfs
apt-get install -y git-lfs
git lfs install

# Track large files
git lfs track "*.npy"
git lfs track "*.npz"
git add .gitattributes
git commit -m "Add Git LFS tracking for large files"
```

---

## Part 7: Using tmux for Long Runs (Recommended)

### 7.1 Create tmux Sessions

```bash
# Session for Jupyter
tmux new -s jupyter
source venv/bin/activate
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
# Detach: Ctrl+B, then D

# Session for monitoring
tmux new -s monitor
watch -n 1 nvidia-smi
# Detach: Ctrl+B, then D

# Session for logs
tmux new -s logs
cd /workspace/action-grounding
tail -f logs/experiment.log
# Detach: Ctrl+B, then D
```

### 7.2 tmux Commands

```bash
# List sessions
tmux ls

# Attach to session
tmux attach -t jupyter

# Switch between sessions
# (when attached): Ctrl+B, then S (shows list)

# Kill session
tmux kill-session -t jupyter
```

---

## Part 8: Alternative - Run as Python Scripts (Headless)

If you prefer to run without Jupyter:

### 8.1 Convert Notebooks to Scripts

```bash
# Install nbconvert if not in requirements
pip install nbconvert

# Convert all notebooks
jupyter nbconvert --to python notebooks/*.ipynb

# Creates:
# notebooks/01_behavioral_phenomenon.py
# notebooks/02_mechanistic_probes.py
# etc.
```

### 8.2 Run Scripts in tmux

```bash
tmux new -s experiment

source venv/bin/activate
cd /workspace/action-grounding

# Run sequentially
python notebooks/01_behavioral_phenomenon.py 2>&1 | tee logs/nb01.log
python notebooks/02_mechanistic_probes.py 2>&1 | tee logs/nb02.log
python notebooks/03_generalization.py 2>&1 | tee logs/nb03.log
python notebooks/04_causal_intervention.py 2>&1 | tee logs/nb04.log

# Detach: Ctrl+B, then D
```

Check progress:
```bash
tail -f logs/nb01.log
```

---

## Part 9: Syncing Results Back to Mac

**Quick Start:**
```bash
# On your Mac, use the sync script:
cd /Users/gadi/Personal/interpret

# Option 1: If you saved connection details:
./scripts/sync_from_pod.sh

# Option 2: Provide IP and port directly:
./scripts/sync_from_pod.sh <POD_IP> <PORT>
ssh root@69.30.85.120 -p 22130 -i ~/.ssh/id_ed25519

# Example:
./scripts/sync_from_pod.sh 123.45.67.89 12345
```

**First time?** Save your connection details:
```bash
./scripts/save_pod_connection.sh
```

**Nothing found to sync?** Check pod status:
```bash
./scripts/check_pod_status.sh <POD_IP> <PORT>
```

### 9.1 Quick Reference: What to Sync

**Always sync:**
- `figures/` - All generated plots and visualizations
- `data/processed/*.parquet` - Processed episode datasets
- `data/processed/*.pkl` - Trained probes and models
- `notebooks/` - Updated notebooks with results
- `logs/` - Experiment logs (optional)

**Large files (sync selectively):**
- `data/processed/*.npy` - Activation arrays (200+ MB each)
- `data/processed/*.npz` - Compressed activations
- `data/labeled/*.npz` - Labeled activation datasets

**Don't sync (already in git or not needed):**
- `venv/` - Virtual environment
- `__pycache__/` - Python cache
- `.env` - API keys (keep on pod only)

### 9.2 Complete Sync Script (Recommended)

Create a sync script on your Mac for easy syncing:

**Create `scripts/sync_from_pod.sh`:**
```bash
#!/bin/bash
# Sync everything from RunPod pod to local Mac
# Usage: ./scripts/sync_from_pod.sh <POD_IP> <PORT>

set -e

POD_IP=${1:-"YOUR_POD_IP"}
PORT=${2:-"YOUR_SSH_PORT"}

if [ "$POD_IP" == "YOUR_POD_IP" ]; then
    echo "Usage: $0 <POD_IP> <PORT>"
    echo "Example: $0 123.45.67.89 12345"
    exit 1
fi

REMOTE_PATH="/workspace/action-grounding"
LOCAL_PATH="/Users/gadi/Personal/interpret"

echo "=================================================="
echo "Syncing from RunPod pod..."
echo "Pod: root@${POD_IP}:${PORT}"
echo "=================================================="

# Create local directories if they don't exist
mkdir -p "${LOCAL_PATH}/figures"
mkdir -p "${LOCAL_PATH}/data/processed"
mkdir -p "${LOCAL_PATH}/data/labeled"
mkdir -p "${LOCAL_PATH}/notebooks"
mkdir -p "${LOCAL_PATH}/logs"

# Sync figures (usually small, fast)
echo ""
echo "[1/6] Syncing figures..."
rsync -avz --progress -e "ssh -p ${PORT}" \
  root@${POD_IP}:${REMOTE_PATH}/figures/ \
  "${LOCAL_PATH}/figures/"

# Sync processed data (parquet, pkl - moderate size)
echo ""
echo "[2/6] Syncing processed data (parquet, pkl)..."
rsync -avz --progress -e "ssh -p ${PORT}" \
  root@${POD_IP}:${REMOTE_PATH}/data/processed/*.parquet \
  "${LOCAL_PATH}/data/processed/" 2>/dev/null || echo "  (No parquet files)"
rsync -avz --progress -e "ssh -p ${PORT}" \
  root@${POD_IP}:${REMOTE_PATH}/data/processed/*.pkl \
  "${LOCAL_PATH}/data/processed/" 2>/dev/null || echo "  (No pkl files)"

# Sync notebooks (to get any updated results/cells)
echo ""
echo "[3/6] Syncing notebooks..."
rsync -avz --progress -e "ssh -p ${PORT}" \
  root@${POD_IP}:${REMOTE_PATH}/notebooks/*.ipynb \
  "${LOCAL_PATH}/notebooks/"

# Sync logs (optional)
echo ""
echo "[4/6] Syncing logs..."
rsync -avz --progress -e "ssh -p ${PORT}" \
  root@${POD_IP}:${REMOTE_PATH}/logs/ \
  "${LOCAL_PATH}/logs/" 2>/dev/null || echo "  (No logs directory)"

# Ask about large files
echo ""
echo "[5/6] Large activation files (.npy, .npz)..."
read -p "Sync large activation files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Syncing .npy files (this may take a while)..."
    rsync -avz --progress -e "ssh -p ${PORT}" \
      root@${POD_IP}:${REMOTE_PATH}/data/processed/*.npy \
      "${LOCAL_PATH}/data/processed/" 2>/dev/null || echo "    (No .npy files)"
    
    echo "  Syncing .npz files..."
    rsync -avz --progress -e "ssh -p ${PORT}" \
      root@${POD_IP}:${REMOTE_PATH}/data/processed/*.npz \
      "${LOCAL_PATH}/data/processed/" 2>/dev/null || echo "    (No .npz files)"
    
    rsync -avz --progress -e "ssh -p ${PORT}" \
      root@${POD_IP}:${REMOTE_PATH}/data/labeled/*.npz \
      "${LOCAL_PATH}/data/labeled/" 2>/dev/null || echo "    (No labeled .npz files)"
else
    echo "  Skipping large files (use manual sync if needed)"
fi

# Sync any other important files
echo ""
echo "[6/6] Syncing other files..."
rsync -avz --progress -e "ssh -p ${PORT}" \
  root@${POD_IP}:${REMOTE_PATH}/data/processed/*.json \
  "${LOCAL_PATH}/data/processed/" 2>/dev/null || echo "  (No json files)"

echo ""
echo "=================================================="
echo "✓ Sync complete!"
echo "=================================================="
echo ""
echo "Synced to: ${LOCAL_PATH}"
echo ""
echo "To sync again, run:"
echo "  ./scripts/sync_from_pod.sh ${POD_IP} ${PORT}"
```

**Make it executable:**
```bash
chmod +x scripts/sync_from_pod.sh
```

**Usage:**
```bash
# From your Mac
cd /Users/gadi/Personal/interpret
./scripts/sync_from_pod.sh <POD_IP> <PORT>

# Example:
./scripts/sync_from_pod.sh 123.45.67.89 12345
```

### 9.3 Manual Sync Commands

If you prefer manual control, use these commands:

**Using rsync (Recommended - faster, resumable):**

```bash
# From your Mac terminal
cd /Users/gadi/Personal/interpret

# Set these variables
POD_IP="your_pod_ip"
PORT="your_ssh_port"

# Sync figures
rsync -avz --progress -e "ssh -p ${PORT}" \
  root@${POD_IP}:/workspace/action-grounding/figures/ \
  ./figures/

# Sync processed data (parquet, pkl)
rsync -avz --progress -e "ssh -p ${PORT}" \
  root@${POD_IP}:/workspace/action-grounding/data/processed/ \
  ./data/processed/

# Sync notebooks
rsync -avz --progress -e "ssh -p ${PORT}" \
  root@${POD_IP}:/workspace/action-grounding/notebooks/*.ipynb \
  ./notebooks/

# Sync large activation files (optional - can be slow)
rsync -avz --progress -e "ssh -p ${PORT}" \
  root@${POD_IP}:/workspace/action-grounding/data/processed/*.npy \
  ./data/processed/
```

**Using scp (Simpler, but slower for large files):**

```bash
# From your Mac terminal
cd /Users/gadi/Personal/interpret

POD_IP="your_pod_ip"
PORT="your_ssh_port"

# Copy figures
scp -P ${PORT} -r root@${POD_IP}:/workspace/action-grounding/figures/* ./figures/

# Copy processed data
scp -P ${PORT} root@${POD_IP}:/workspace/action-grounding/data/processed/*.parquet ./data/processed/
scp -P ${PORT} root@${POD_IP}:/workspace/action-grounding/data/processed/*.pkl ./data/processed/

# Copy notebooks
scp -P ${PORT} root@${POD_IP}:/workspace/action-grounding/notebooks/*.ipynb ./notebooks/
```

### 9.4 Selective Sync (Large Files Only)

If you only need specific large files:

```bash
# Sync a specific activation file
rsync -avz --progress -e "ssh -p ${PORT}" \
  root@${POD_IP}:/workspace/action-grounding/data/processed/activations.npy \
  ./data/processed/

# Sync all labeled activations
rsync -avz --progress -e "ssh -p ${PORT}" \
  root@${POD_IP}:/workspace/action-grounding/data/labeled/ \
  ./data/labeled/
```

### 9.5 Using Git (For Code Changes Only)

**For code changes (not data files):**

```bash
# On pod
cd /workspace/action-grounding
git add src/ notebooks/*.ipynb config.yaml  # code changes only
git commit -m "Update notebooks with results"
git push

# On Mac
cd /Users/gadi/Personal/interpret
git pull
```

**Note:** Don't commit large data files (.npy, .npz) - use rsync/scp instead.

### 9.6 Verify Sync

After syncing, verify files are present:

```bash
# On Mac
cd /Users/gadi/Personal/interpret

# Check figures
ls -lh figures/
# Should see: *.png, *.pdf files

# Check processed data
ls -lh data/processed/
# Should see: *.parquet, *.pkl files

# Check notebooks
ls -lh notebooks/*.ipynb
# Should see updated notebooks
```

### 9.7 Troubleshooting Sync

**Issue: Connection timeout**
```bash
# Use rsync with keepalive
rsync -avz --progress -e "ssh -p ${PORT} -o ServerAliveInterval=60" \
  root@${POD_IP}:/workspace/action-grounding/figures/ \
  ./figures/
```

**Issue: Permission denied**
```bash
# Make sure you're using the correct user (usually 'root' on RunPod)
# Check: ssh root@${POD_IP} -p ${PORT}
```

**Issue: Large file transfer interrupted**
```bash
# rsync automatically resumes, but you can also use:
rsync -avz --partial --progress -e "ssh -p ${PORT}" \
  root@${POD_IP}:/workspace/action-grounding/data/processed/large_file.npy \
  ./data/processed/
# The --partial flag keeps partial transfers
```

**Issue: rsync command not found**
```
bash: line 1: rsync: command not found
rsync: connection unexpectedly closed
```

**Fix:** The sync script will automatically fall back to `scp` if `rsync` isn't installed. To install `rsync` on the pod for better performance:

```bash
# SSH into pod
ssh root@<POD_IP> -p <PORT>

# Install rsync
apt-get update
apt-get install -y rsync
```

Or run the setup script which installs it automatically:
```bash
./scripts/setup_runpod.sh
```

The sync script will detect if rsync is available and use it, otherwise it falls back to scp.

**Issue: Nothing found to sync (all directories/files missing)**

If sync script shows "(No figures directory)", "(No parquet files)", etc.:

1. **Check pod status first:**
```bash
# Diagnose what's on the pod
./scripts/check_pod_status.sh <POD_IP> <PORT>
```

2. **Common causes:**

   **A. Repository not cloned yet:**
   ```bash
   # SSH into pod
   ssh root@<POD_IP> -p <PORT>
   
   # Clone repository
   cd /workspace
   git clone https://github.com/YOUR_USERNAME/action-grounding.git
   cd action-grounding
   
   # Run setup
   ./scripts/setup_runpod.sh
   ```

   **B. No work done yet:**
   - This is normal if you just set up the pod
   - Run notebooks first to generate data/figures
   - Then sync will find files

   **C. Wrong path:**
   ```bash
   # Check what's actually on the pod
   ssh root@<POD_IP> -p <PORT>
   ls -la /workspace/
   # Verify the directory structure
   ```

   **D. Using SSH key:**
   If you're using an SSH key (like `-i ~/.ssh/id_ed25519`), update the sync script:
   ```bash
   # Edit sync_from_pod.sh and add SSH key option
   # Or use SSH config file (~/.ssh/config):
   Host runpod
       HostName <POD_IP>
       Port <PORT>
       User root
       IdentityFile ~/.ssh/id_ed25519
   
   # Then use: ssh runpod
   ```

---

## Part 10: RunPod CLI (Advanced)

### 10.1 Install RunPod CLI

```bash
# On your Mac
pip install runpod

# Login
runpod config
# Enter your API key from runpod.io/console/user/settings
```

### 10.2 CLI Commands

```bash
# List pods
runpod pod list

# Get pod details
runpod pod get <POD_ID>

# Stop pod (saves money)
runpod pod stop <POD_ID>

# Start pod
runpod pod start <POD_ID>

# SSH into pod
runpod pod ssh <POD_ID>
```

### 10.3 Automated Deployment Script

I'll create this for you next...

---

## Part 11: Cost Management

### 11.1 Estimated Costs

| Phase | Time | GPU | Cost (RTX 4090 @ $0.40/hr) |
|-------|------|-----|----------------------------|
| Setup | 30 min | Idle | $0.20 |
| Notebook 01 | 3 hours | 90% | $1.20 |
| Notebook 02 | 1.5 hours | 95% | $0.60 |
| Notebook 03 | 30 min | 80% | $0.20 |
| Notebook 04 | 2 hours | 90% | $0.80 |
| **Total** | **~7.5 hours** | - | **~$3.00** |

### 11.2 Minimize Costs

1. **Use persistent volumes** - Don't lose work when stopping
2. **Stop pod between sessions:**
   ```bash
   # When done for the day
   tmux kill-server  # Kill all tmux sessions
   exit  # Exit SSH
   # Then in RunPod UI: Stop pod
   ```
3. **Resume next day:**
   - Start pod in UI
   - SSH in
   - `cd /workspace/action-grounding && source venv/bin/activate`
   - `tmux attach -t jupyter` or restart Jupyter

---

## Part 12: Execution Timeline

### Recommended Schedule

**Session 1: Initial Setup + Notebook 01 (4 hours)**
```bash
0:00 - Setup environment (30 min)
0:30 - Test small run (30 min)
1:00 - Start Notebook 01 full run (2.5 hours)
3:30 - Verify results, save checkpoint
```

**Session 2: Notebook 02 (2 hours)**
```bash
0:00 - Resume, verify episodes.parquet
0:05 - Start Notebook 02 (1.5 hours)
1:35 - Verify probes, check position analysis
1:50 - Save checkpoint
```

**Session 3: Notebooks 03 + 04 (3 hours)**
```bash
0:00 - Resume, run Notebook 03 (30 min)
0:30 - Verify transfer results
0:35 - Start Notebook 04 (2 hours)
2:35 - Verify steering results
2:45 - Sync all results back to Mac
```

---

## Part 13: Quick Start Script

I'll create an automated setup script for you:
