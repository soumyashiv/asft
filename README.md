# ASFT: Training Acceleration Framework

**ASFT** is a production-grade, enterprise-ready Training Acceleration Framework designed to dramatically reduce the resources required to train and deploy AI systems. 

By strategically avoiding unnecessary fine-tuning and applying state-of-the-art acceleration techniques when training is required, ASFT reduces compute time, GPU costs, energy consumption, and dataset requirements — all while maintaining or improving model capability and accuracy.

---

## The "Train Only If Necessary" Philosophy

ASFT operates as a **decision engine first, and a training framework second**. Before allocating any compute budget to fine-tuning, the framework systematically evaluates zero-shot and retrieval-based alternatives.

The `AutoOptimizer` subsystem evaluates tasks in the following strict hierarchy:
1. **Working Memory:** Is the answer in the immediate context window?
2. **Episodic Memory / RAG:** Can the model answer correctly by retrieving facts via FTS5/Vector search?
3. **Skill Packs:** Can a specialized function or programmatic skill handle the reasoning?
4. **Knowledge Distillation:** Can we transfer capability from a larger teacher without full training?
5. **Parameter-Efficient Fine-Tuning (QLoRA):** Optimize using 4-bit quantization and low-rank adapters.
6. **Full Fine-Tuning:** The absolute last resort, triggered only if ROI justifies the massive compute cost.

## Core Acceleration Subsystems

When training is deemed necessary, ASFT employs a suite of research-backed optimization subsystems:

### 1. Cost & ROI Estimation
Uses Kaplan (2020) and Chinchilla (2022) scaling laws to project exact GPU-hours and USD costs *before* any compute is allocated. Training jobs are automatically rejected if the expected accuracy gain does not justify the projected cost.

### 2. Adaptive Sample Selection (Data Pruning)
Uses Perplexity and EL2N (Error L2-Norm) scoring to evaluate dataset quality. Automatically prunes redundant, noisy, or "easy" samples from the dataset, often reducing required dataset sizes by 50%–80% without degrading final accuracy.

### 3. Dynamic Sparse Training (RigL)
Implements Rigged Lottery (RigL) dynamic sparsity. Instead of updating dense matrices, the framework adaptively prunes low-magnitude weights and grows new connections based on gradient signals, drastically reducing parameter updates.

### 4. Knowledge Distillation (Hinton 2015)
Uses KL-Divergence and Temperature-scaled soft targets to transfer reasoning patterns from massive teacher models to smaller, efficient student models, vastly accelerating convergence compared to standard label-based training.

### 5. Catastrophic Forgetting Mitigation (EWC)
Implements Elastic Weight Consolidation (Kirkpatrick 2017) to approximate the Fisher Information matrix. Important weights from previous tasks are protected by a quadratic penalty, allowing continual learning without destroying prior capabilities.

---

## Architecture & Security

ASFT is designed for robust enterprise deployment with zero trust.

* **Zero-Execution Verification:** The framework's accuracy verification layers never execute LLM-generated code. Code validation uses strictly AST-based parsing (`RestrictedPython`), and mathematical verification relies exclusively on the SymPy Computer Algebra System.
* **Bounded Persistent Memory:** Fast, O(1) semantic lookups via SQLite FTS5 inverted indices, replacing costly linear scans.
* **Memory-Safe Work Queues:** API server delegates intensive GPU compute to sandboxed isolated processes via `ProcessPoolExecutor`, guaranteeing the main event loop is never blocked.

---

## Installation

```bash
# Python 3.10+ required
pip install -e .

# Optional extras
pip install -e ".[faiss]"   # For CPU vector search
pip install -e ".[faiss-gpu]" # For GPU vector search
pip install -e ".[viz]"     # For analytical plotting (Plotly/Matplotlib)
```

## Quickstart

### 1. Estimate Training Cost
```python
from asft.optimizer.cost_estimator import CostEstimator

estimator = CostEstimator()
projection = estimator.estimate(
    model_name="Qwen/Qwen2-7B",
    dataset_size=50_000,
    method="qlora"
)

print(f"Estimated Cost: ${projection.cost_usd:.2f}")
print(f"GPU Hours: {projection.gpu_hours:.2f}")
```

### 2. Run Auto-Optimization Decision
```python
from asft.optimizer.auto_optimizer import AutoOptimizer

optimizer = AutoOptimizer()
decision = optimizer.decide(
    task="Provide medical triage recommendations based on symptoms",
    domain="medical",
    target_accuracy=0.92,
    budget_usd=50.0
)

print(f"Action: {decision.action}")
print(f"Reasoning: {decision.reasoning}")
```

---

## Status

**Current Version:** `0.3.0` (Production Ready)  
**Security Posture:** Hardened (AST-only code verification, SymPy CAS math evaluation)

> ⚠️ **Note:** The legacy gradient-masking `SparseTrainer` has been officially deprecated as it did not yield actual FLOP/memory reductions on dense hardware. It has been replaced by the `DynamicSparseTrainer` (RigL) and `ParameterSelector`.
