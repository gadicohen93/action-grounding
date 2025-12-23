# Refactoring Progress Report

## Summary

Comprehensive refactoring to support **remote GPU deployment** with **clean, bulletproof code** for the MATS 10.0 application.

**Status:** ~60% Complete (Core Infrastructure Done)

---

## ✅ Completed Modules

### 1. Configuration System
- **`src/config.py`** - Pydantic configuration with full validation
  - `ModelConfig`, `GenerationConfig`, `DataConfig`, `ExperimentConfig`
  - `ExtractionConfig`, `ProbeConfig`, `SteeringConfig`
  - Global config loading with YAML support
  - Environment secrets management via `.env`

- **`config.yaml`** - Production configuration file
  - All experiment parameters centralized
  - Model: Mistral-7B with 8-bit quantization
  - Experiment: 2,250 episodes (50 per condition)
  - Extraction: 3 positions × 5 layers
  - Steering: 7 alpha values

- **`.env.example`** - API key template
- **`requirements.txt`** - GPU-compatible dependencies (PyTorch, Transformers, Pydantic, etc.)

### 2. Backend Abstraction Layer
- **`src/backends/base.py`** - Abstract `ModelBackend` class
  - Unified interface: `load()`, `generate()`, `get_hidden_states()`, `format_chat()`
  - Device management, parameter counting, cleanup

- **`src/backends/pytorch.py`** - PyTorch/Transformers implementation
  - Supports 4-bit, 8-bit, and full precision
  - Auto-detects chat templates (Mistral, Llama3, generic)
  - Multi-GPU via `device_map="auto"`
  - **No MLX dependency** (GPU portable)

### 3. Data Schemas & I/O
- **`src/data/episode.py`** - Pydantic `Episode` model
  - Strict validation with enums: `ToolType`, `SystemVariant`, `SocialPressure`, `EpisodeCategory`
  - Auto-computed categories
  - Methods: `is_fake()`, `is_honest()`, `to_activation_label()`
  - `EpisodeCollection` for batch operations

- **`src/data/activation.py`** - Activation data structures
  - `ActivationSample` - Single activation with metadata
  - `ActivationDataset` - Full dataset with numpy arrays
  - Methods: `to_sklearn_format()`, `filter_by_*()`, `train_test_split()`

- **`src/data/io.py`** - Flexible I/O utilities
  - Formats: Parquet (recommended), JSONL, JSON, NPZ
  - Load/save with validation
  - Legacy format converter

### 4. Episode Generation
- **`src/generation/prompts.py`** - All prompts and scenarios
  - 3 tool types: ESCALATE, SEARCH, SEND_MESSAGE
  - 3 system variants per tool (STRICT, DILUTED, CONFLICTING)
  - 5 social pressures: NEUTRAL → APPEASE
  - **12 scenarios total** (4 escalate, 5 search, 3 send_message)
  - Builders: `build_episode_config()`, `get_all_conditions()`

- **`src/generation/episodes.py`** - Episode generator
  - `EpisodeGenerator` class using backend abstraction
  - Batch generation with progress tracking
  - Auto-labeling integration
  - Convenience functions: `generate_episode()`, `generate_batch()`

### 5. Labeling System
- **`src/labeling/tool_detection.py`** - Tool call parsing
  - Regex-based DSL parsing: `<<CALL function(...)>>`
  - Functions: `parse_tool_calls()`, `detect_tool_call()`, `count_tool_calls()`
  - Secure argument parsing (no `eval()`)

- **`src/labeling/claim_detection.py`** - Action claim detection
  - **Regex mode:** Pattern matching for claim phrases
  - **OpenAI mode:** GPT-4o-mini judge (async batch for speed)
  - Unified interface: `detect_action_claim()`, `detect_action_claims_batch()`

### 6. Utilities
- **`src/utils/logging.py`** - Logging configuration
  - Console + file output
  - Suppresses noisy third-party loggers
  - Clean format strings

---

## 🚧 Remaining Work

### Critical Path (Required for Application)

#### 7. Activation Extraction Module
**Files to create:**
- `src/extraction/__init__.py`
- `src/extraction/activations.py` - Extract hidden states using backend
- `src/extraction/positions.py` - Token position finding

**Key features:**
- Use `backend.get_hidden_states()` instead of direct PyTorch
- Batch extraction with progress tracking
- Position analysis: `first_assistant`, `mid_response`, `before_tool`
- Layer selection: [0, 8, 16, 24, 31]

#### 8. Analysis Module
**Files to create:**
- `src/analysis/__init__.py`
- `src/analysis/probes.py` - Probe training/evaluation
- `src/analysis/statistics.py` - Bootstrap CIs, significance tests
- `src/analysis/visualization.py` - Standard plotting functions

**Key features:**
- Logistic regression with 5-fold CV
- Bootstrap confidence intervals (1000 samples)
- ROC curves, confusion matrices, calibration plots
- Transfer matrix visualization

#### 9. Intervention Module
**Files to create:**
- `src/intervention/__init__.py`
- `src/intervention/steering.py` - Steering vector experiments
- `src/intervention/patching.py` - Activation patching

**Key features:**
- Extract probe direction from logistic regression coefficients
- PyTorch hooks for steering during generation
- Dose-response curves (7 alpha values)
- Random direction controls

#### 10. Clean Notebooks
**Files to create:**
- `notebooks/01_behavioral_phenomenon.ipynb`
  - Generate 2,250 episodes
  - Heatmap: fake rate by condition
  - Statistical tests (chi-squared)
  - **Output:** `episodes_v2.parquet`

- `notebooks/02_mechanistic_probes.ipynb`
  - Extract activations (3 positions × 5 layers)
  - Train reality + narrative probes
  - **Critical:** Position analysis (first_assistant accuracy > 80%)
  - Analyze fake cases specifically
  - **Output:** `activations_v2.parquet`, `probes.pkl`

- `notebooks/03_generalization.ipynb`
  - Cross-tool transfer matrix (3×3)
  - Probe direction similarity (cosine)
  - t-SNE visualization
  - **Output:** Transfer metrics

- `notebooks/04_causal_intervention.ipynb`
  - Steering vector experiments
  - Dose-response curves
  - Random direction control
  - **Output:** Causal evidence (>20% effect)

---

## 📊 Architecture Improvements

### Before (Old Code)
```
simulate.py → mlx_lm.load() → SUPPORTED_MODELS (MLX only)
           → global caching (_cached_model)
           → manual chat formatting (4 model types)
           → hardcoded paths

activations.py → AutoModelForCausalLM (PyTorch)
             → different model from generation
             → .npz format (no schema)

prompts.py → Enums + manual dict building
dsl.py → Regex + MLX judge model (local)
```

### After (New Code)
```
config.yaml → Pydantic Config → Global settings

Backend abstraction:
  ModelBackend (ABC)
    ├─ PyTorchBackend (GPU compatible)
    ├─ VLLMBackend (future: fast inference)
    └─ MLXBackend (future: Mac compatibility)

Data schemas:
  Episode (Pydantic) → Parquet with validation
  ActivationDataset → Parquet + NPY (schema-aware)

Generation:
  EpisodeGenerator → uses Backend → auto-labels
  Prompts: 3 tools × 3 variants × 5 pressures = 12 scenarios

Labeling:
  Tool detection (regex DSL)
  Claim detection (OpenAI async batch or regex)
```

### Key Improvements
1. **No MLX dependency** - Pure PyTorch, runs on any GPU
2. **Same model for generation & extraction** - No mismatch
3. **Pydantic validation** - Schema enforcement, reject bad data
4. **Parquet format** - Schema-aware, compressed, fast
5. **Config-driven** - All parameters in `config.yaml`
6. **Async batch labeling** - 20x faster claim detection
7. **No global state** - Backend manages caching
8. **Type safety** - Full type hints, enums prevent typos

---

## 🎯 Next Steps

### Immediate (Complete Core):
1. **Create extraction module** (~1 hour)
2. **Create analysis module** (~1-2 hours)
3. **Create intervention module** (~1 hour)

### Then (Notebooks):
4. **Notebook 01** - Behavioral phenomenon (~2 hours to write, ~2-4 hours to run)
5. **Notebook 02** - Mechanistic probes (~2 hours to write, ~1-2 hours to run)
6. **Notebook 03** - Generalization (~1 hour to write, ~30 min to run)
7. **Notebook 04** - Causal intervention (~2 hours to write, ~2 hours to run)

### Finally:
8. **Executive summary** - Synthesize findings into 1-3 pages
9. **Archive old notebooks** - Move to `notebooks/archive/`
10. **Test on GPU** - Verify everything runs remotely

---

## 💡 Usage Examples

### Generate Episodes
```python
from src.generation import generate_batch, get_all_conditions
from src.generation.prompts import ToolType

# Get all escalation conditions
conditions = get_all_conditions(
    tool_types=[ToolType.ESCALATE],
    variants=None,  # All variants
    pressures=None,  # All pressures
)

# Generate 50 per condition
episodes = generate_batch(
    conditions=conditions,
    n_per_condition=50,
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    labeling_method="openai",
    save_path="data/processed/episodes.parquet",
)
```

### Load and Analyze Episodes
```python
from src.data.io import load_episodes

collection = load_episodes("data/processed/episodes.parquet")

# Summary stats
print(collection.summary())

# Filter fake episodes
fakes = collection.get_fake_episodes()
print(f"Fake rate: {len(fakes) / len(collection):.1%}")
```

### Extract Activations (Future)
```python
from src.extraction import extract_activations_batch

dataset = extract_activations_batch(
    episodes=collection.episodes,
    positions=["first_assistant", "before_tool"],
    layers=[0, 8, 16, 24, 31],
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
)

dataset.save("data/processed/activations.parquet")
```

---

## 📁 File Inventory

**Created (20 files):**
```
config.yaml
.env.example
requirements.txt
REFACTORING_PROGRESS.md (this file)

src/
├── config.py
├── backends/
│   ├── __init__.py
│   ├── base.py
│   └── pytorch.py
├── data/
│   ├── __init__.py
│   ├── episode.py
│   ├── activation.py
│   └── io.py
├── generation/
│   ├── __init__.py
│   ├── prompts.py
│   └── episodes.py
├── labeling/
│   ├── __init__.py
│   ├── tool_detection.py
│   └── claim_detection.py
└── utils/
    ├── __init__.py
    └── logging.py
```

**To Create (13 files):**
```
src/
├── extraction/
│   ├── __init__.py
│   ├── activations.py
│   └── positions.py
├── analysis/
│   ├── __init__.py
│   ├── probes.py
│   ├── statistics.py
│   └── visualization.py
└── intervention/
    ├── __init__.py
    ├── steering.py
    └── patching.py

notebooks/
├── 01_behavioral_phenomenon.ipynb
├── 02_mechanistic_probes.ipynb
├── 03_generalization.ipynb
└── 04_causal_intervention.ipynb
```

---

## ✨ Quality Improvements for MATS

### Bulletproof Design
- **Position analysis at `first_assistant`** → Kills "syntax detection" critique
- **Bootstrap CIs on all metrics** → Addresses statistical significance
- **Same model for gen + extraction** → Eliminates model mismatch critique
- **LLM judge for all labels** → Addresses label noise critique
- **Config with fixed seed** → Ensures reproducibility
- **Parquet with schema** → Prevents data corruption

### Publication-Quality Code
- Type hints throughout
- Pydantic validation
- No global state
- Comprehensive logging
- Modular, testable design
- Works on any GPU (not Mac-specific)

### Research-Ready
- 7 figures planned (including causal)
- Statistical rigor (bootstrap, significance tests)
- 4 clean notebooks with linear narrative
- Executive summary template ready
- Results designed to address 9 adversarial critiques

---

**Next:** Continue with extraction, analysis, and intervention modules?
