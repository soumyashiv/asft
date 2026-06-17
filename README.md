# ASFT — Adaptive Sparse Fine-Tuning Framework

<div align="center">

**Next-generation hardware-adaptive AI learning framework**  
*Dramatically reduces training time and resource consumption while improving accuracy*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com)

</div>

---

## What is ASFT?

ASFT implements a **6-tier learning hierarchy** that prioritizes fast, cheap strategies before resorting to expensive retraining:

```
Memory Retrieval           ← fastest, free
  → Workflow Optimization  ← no training required
    → Tool Learning        ← tool-use, no weight updates
      → Skill Packs        ← task-specific LoRA adapters
        → Sparse Fine-Tuning ← update 1–5% of parameters
          → Full Fine-Tuning ← last resort only
```

**Full retraining is always the last resort.**

---

## Key Features

| Feature | Description |
|---|---|
| **Hardware Auto-Detection** | Profiles CPU/GPU/RAM and auto-selects precision, quantization, batch size |
| **Sparse Training** | Updates only 1–5% of critical parameters (gradient + activation-guided selection) |
| **LoRA / QLoRA** | PEFT adapter integration with 4-bit/8-bit quantization support |
| **5-Tier Memory** | Working → Episodic → Semantic → Long-Term → Vector (ChromaDB/FAISS/Qdrant) |
| **6 Skill Packs** | Coding, Research, Planning, Mathematics, Trading, Automation |
| **Smart Routing** | Keyword + embedding + performance-history routing to best expert |
| **Dataset Compression** | MinHash dedup → KMeans clustering → representative selection (30–70% reduction) |
| **Accuracy Engine** | Multi-pass reasoning, self-critique, verification, confidence scoring |
| **Self-Improvement** | Analyzes failures → optimizes prompts/workflows before retraining |
| **Evolutionary Optimizer** | Gradient-free mutation+selection for prompt and strategy optimization |
| **REST API** | Full FastAPI server with training, memory, skills, benchmark endpoints |
| **CLI** | Rich Typer CLI with `init`, `train`, `compress`, `skill`, `memory`, `api` |

---

## Quick Start

### Install

```bash
# Clone or install
git clone https://github.com/your-org/asft
cd asft
pip install -e ".[dev]"
```

### Initialize workspace

```bash
asft init
```

This detects your hardware and writes a config file:
```
✓ Config saved: ./asft_config.yaml
✓ Data dir   : ./asft_data
✓ Recommended: asft with bf16
```

### Check system status

```bash
asft status
```

### Route a task to the best skill

```bash
asft skill route "Write a Python function to parse JSON"
# → Routed to: ['coding'] (score: 0.867)
```

### Train a model

```bash
asft train \
  --model Qwen/Qwen2-0.5B \
  --dataset ./data/train.jsonl \
  --method asft \
  --steps 500 \
  --sparsity 0.95
```

### Compress a dataset

```bash
asft compress --dataset ./data/train.jsonl --output compressed
# Original: 10000 samples → Final: 3200 samples (68% reduction)
```

### Start the API server

```bash
asft api --port 8080
# → http://localhost:8080/docs (Swagger UI)
```

---

## Architecture

```
asft/
├── core/                    # Hardware profiler, config, registry
│   ├── hardware_profiler.py # GPU/CPU/RAM detection + recommendations
│   ├── config.py            # Central Pydantic Settings config
│   └── registry.py          # Thread-safe component registry
│
├── memory/                  # 5-tier memory architecture
│   ├── working_memory.py    # Fast in-session TTL scratch space
│   ├── episodic_memory.py   # SQLite event store with temporal indexing
│   ├── semantic_memory.py   # Fact/concept triple store
│   ├── long_term_memory.py  # Consolidated durable knowledge
│   ├── vector_memory.py     # ChromaDB/FAISS/Qdrant adapter layer
│   ├── consolidator.py      # Episodic → long-term consolidation
│   └── memory_manager.py   # Unified memory facade
│
├── sparse/                  # Sparse fine-tuning engine
│   ├── activation_analyzer.py # Hook-based activation collection
│   ├── neuron_selector.py   # Importance scoring + sparse mask
│   ├── sparse_trainer.py    # Training loop with delta checkpointing
│   └── lora_adapter.py      # LoRA/QLoRA PEFT integration
│
├── skills/                  # Modular skill pack system
│   ├── skill_pack.py        # Base class + MergedSkillPack
│   ├── skill_router.py      # Keyword + embedding + perf routing
│   └── packs/               # 6 built-in skill packs
│       ├── coding.py        # Code generation and debugging
│       ├── research.py      # Information synthesis and analysis
│       ├── planning.py      # Project planning and decomposition
│       ├── mathematics.py   # Computation and symbolic reasoning
│       ├── trading.py       # Market analysis and signals
│       └── automation.py    # Workflow and pipeline automation
│
├── dataset/                 # Dataset compression pipeline
│   ├── deduplicator.py      # MinHash LSH near-duplicate removal
│   ├── clusterer.py         # KMeans/DBSCAN semantic clustering
│   ├── representative_selector.py  # Centroid/diversity selection
│   └── compressor.py        # Full pipeline orchestrator
│
├── accuracy/                # Accuracy enhancement subsystem
│   ├── confidence_scorer.py # Multi-dim output quality scoring
│   ├── multi_pass_reasoner.py # Best-of-K / self-consistency
│   ├── self_critique.py     # Issue detection and revision
│   └── verification_layer.py # Math/code/memory cross-checking
│
├── improvement/             # Self-improvement engine
│   └── self_improvement_engine.py  # Task analysis, prompt/workflow optimization
│
├── evolutionary/            # Gradient-free optimization
│   └── evolutionary_engine.py  # Mutation, crossover, fitness evaluation
│
├── hardware/                # Hardware adaptation
│   └── optimizer.py         # Quantizer, offloader, batch scheduler
│
├── layers/                  # Dynamic layer selection
│   └── layer_selection.py   # Benchmarker, selector, explainability
│
├── benchmark/               # Benchmarking suite
│   ├── runner.py            # Inference timing, memory, accuracy
│   └── reporter.py          # HTML + JSON report generation
│
├── api/                     # REST API
│   └── server.py            # FastAPI with all endpoints
│
└── cli/                     # Command-line interface
    └── main.py              # Typer CLI with all commands
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/status` | System health, hardware, memory stats |
| GET | `/hardware` | Hardware profile and recommendations |
| POST | `/train` | Launch a training job (async) |
| GET | `/train/{job_id}` | Get training job status |
| GET | `/skills` | List all registered skill packs |
| POST | `/skills/route` | Route a task to best skill |
| POST | `/skills/{name}/process` | Execute a skill on a task |
| POST | `/memory/query` | Query the memory system |
| POST | `/memory/facts` | Store a semantic fact |
| GET | `/memory/stats` | Memory system statistics |
| POST | `/dataset/compress` | Compress a dataset (async) |
| GET | `/benchmark/results` | Latest benchmark results |

---

## Training Methods

| Method | Params Updated | Memory Use | Speed | When to Use |
|---|---|---|---|---|
| **full** | 100% | Very High | Slow | Large dataset, high accuracy goal |
| **lora** | 1–10% | Low | Fast | Moderate adaptation |
| **qlora** | 1–10% | Very Low | Fast | Limited VRAM |
| **sparse** | 1–5% | Low | Fast | Critical layer selection |
| **asft** | 1–5% + memory | Low | Fastest | Default — all hardware |

---

## Vector Database Backends

ASFT defaults to ChromaDB but supports switching via config alone:

```yaml
# asft_config.yaml
vector:
  backend: chromadb    # Options: chromadb | faiss | qdrant
  chromadb_path: ./asft_data/chroma
  qdrant_url: http://localhost:6333
```

---

## Running Examples

```bash
# No GPU required
python examples/01_hardware_and_routing.py
python examples/02_memory_systems.py
python examples/03_accuracy_and_dataset.py
```

---

## Running Tests

```bash
pytest tests/ -v                    # Full suite
pytest tests/ -v -k "Memory"        # Memory tests only
pytest tests/ -v -k "Skill"         # Skill pack tests
pytest tests/ -v --tb=short         # Short traceback
```

---

## Configuration Reference

```yaml
# asft_config.yaml
data_dir: ./asft_data

hardware:
  precision: bf16          # fp32 | fp16 | bf16
  quantization: 4bit       # none | 4bit | 8bit
  max_vram_gb: 0           # 0 = auto
  batch_size: 1

model:
  name: Qwen/Qwen2-0.5B
  cache_dir: ./asft_data/models
  trust_remote_code: true

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05

sparse:
  sparsity_ratio: 0.95
  max_steps: 1000
  learning_rate: 2.0e-4
  gradient_accumulation_steps: 4
  eval_steps: 50
  save_steps: 100
  delta_output_dir: ./asft_data/deltas

vector:
  backend: chromadb
  embedding_model: all-MiniLM-L6-v2

memory:
  episodic_db: ./asft_data/memory/episodic.db
  semantic_db: ./asft_data/memory/semantic.db
  long_term_db: ./asft_data/memory/long_term.db
  consolidation_interval_hours: 24
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Built with ❤️ — soumyashiv : Adaptive Sparse Fine-Tuning Framework*
