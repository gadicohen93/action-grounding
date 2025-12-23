# RunPod Deployment Guide: Step-by-Step

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

Once running, click your pod and note:

- **SSH command:** `ssh root@X.X.X.X -p XXXXX`
- **Jupyter URL:** Usually shown as "Connect" button
- **Pod ID:** You'll need this

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
```

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

### 9.1 Using scp

```bash
# From your Mac terminal (not SSH)
cd /Users/gadi/Personal/interpret

# Copy all figures
scp -P <PORT> root@<POD_IP>:/workspace/action-grounding/figures/* ./figures/

# Copy data files
scp -P <PORT> root@<POD_IP>:/workspace/action-grounding/data/processed/*.parquet ./data/processed/

# Copy trained probes
scp -P <PORT> root@<POD_IP>:/workspace/action-grounding/data/processed/*.pkl ./data/processed/
```

### 9.2 Using rsync (Better for Large Files)

```bash
# From Mac
rsync -avz -e "ssh -p <PORT>" \
  root@<POD_IP>:/workspace/action-grounding/figures/ \
  ./figures/

rsync -avz -e "ssh -p <PORT>" \
  root@<POD_IP>:/workspace/action-grounding/data/processed/ \
  ./data/processed/
```

### 9.3 Using Git (For Code Changes)

```bash
# On pod
cd /workspace/action-grounding
git add results/ figures/  # if you want to version these
git commit -m "GPU run results"
git push

# On Mac
git pull
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
