"""
ASFT Example 3 — Dataset Compression (no GPU required)
=======================================================
Demonstrates the full dedup → cluster → select pipeline on a synthetic dataset.
"""

import json
import os
import tempfile

print("=" * 60)
print("ASFT — Example 3: Dataset Compression")
print("=" * 60)

# Create a synthetic dataset with duplicates and redundancy
SAMPLES = [
    "Python is a programming language designed for readability and simplicity.",
    "Python is a language that prioritizes code readability and clean syntax.",  # near-dup
    "Java is a statically typed, object-oriented programming language.",
    "Machine learning involves training models on data to make predictions.",
    "Deep learning is a subset of ML using neural networks with many layers.",
    "Neural networks are computational models inspired by the human brain.",
    "The Transformer architecture uses attention mechanisms for sequence modeling.",
    "BERT uses bidirectional transformers for natural language understanding tasks.",
    "GPT models use causal language modeling for text generation tasks.",
    "Fine-tuning adapts a pre-trained model to a specific downstream task.",
    "LoRA reduces trainable parameters by adding low-rank matrices to layers.",
    "QLoRA combines quantization and LoRA for memory-efficient fine-tuning.",
    "Sparse training only updates the most important parameters of a model.",
    "Sparse fine-tuning selectively updates the most critical neural parameters.",  # near-dup
    "Dataset quality matters more than dataset size for fine-tuning outcomes.",
    "Deduplication removes near-duplicate samples to improve training efficiency.",
    "Clustering groups similar samples to identify redundant training examples.",
    "Representative selection picks the most informative samples from each group.",
    "ASFT combines sparse training, memory retrieval, and self-improvement.",
    "Adaptive learning adjusts the training strategy based on hardware constraints.",
]

print(f"\nOriginal dataset: {len(SAMPLES)} samples")

# Write to temp JSONL
with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
    for text in SAMPLES:
        f.write(json.dumps({"text": text}) + "\n")
    tmp_path = f.name

print(f"Written to: {tmp_path}")

# Run compression (without clustering to avoid sentence-transformers dependency)
print("\n[1] Running deduplication...")
from asft.dataset.deduplicator import DatasetDeduplicator

try:
    deduper = DatasetDeduplicator(threshold=0.82, num_perm=64)
    texts = SAMPLES.copy()
    ids = [str(i) for i in range(len(texts))]
    kept_texts, kept_ids, dedup_stats = deduper.deduplicate(texts, ids)
    print(f"  Original: {dedup_stats['original_count']}")
    print(f"  After dedup: {dedup_stats['kept_count']}")
    print(
        f"  Removed: {dedup_stats['removed_count']} ({dedup_stats['reduction_ratio']:.1%} reduction)"
    )
except ImportError:
    print("  datasketch not installed — skipping dedup (pip install datasketch)")
    kept_texts = SAMPLES

print("\n[2] Confidence scoring on skill outputs...")
from asft.accuracy.confidence_scorer import ConfidenceScorer

scorer = ConfidenceScorer()
test_outputs = [
    "The answer is definitely X.",
    "I think maybe the result could possibly be around X.",
    "```python\ndef solve(n):\n    return n * 2\n```\nResult: X",
    "According to recent studies, experts agree that X is widely known.",
]
for o in test_outputs:
    s = scorer.score(o)
    print(f"  [{s.label:>6}] {s.composite:.3f} — {o[:50]}...")

print("\n[3] Multi-pass reasoning demo (no model)...")
import random

from asft.accuracy.multi_pass_reasoner import MultiPassReasoner


def mock_generate(n: int):
    templates = [
        "The derivative of f(x) = 3x^3 is f'(x) = 9x^2.",
        "f'(x) = 9x^2 by the power rule, applied to each term.",
        "Using the power rule: f'(x) = 9x^2 + 4x - 5.",
    ]
    return random.choices(templates, k=n)


reasoner = MultiPassReasoner(k=3, strategy="best_of_k")
result = reasoner.reason(mock_generate, task_type="mathematics")
print(f"  Best output: {result.best_output}")
print(f"  Confidence : {result.best_score.composite:.3f}")
print(f"  Passes used: {result.passes_used}")

print("\n[4] Self-critique demo (no model)...")
from asft.accuracy.self_critique import SelfCritiqueEngine

critic = SelfCritiqueEngine(max_rounds=1)
bad_output = "This is definitely proven by all experts. The answer is X. Therefore the result is Y."
result = critic.critique(bad_output, original_task="Solve X", generate_fn=None)
print(f"  Issues found : {result.issues_found}")
print(f"  Was revised  : {result.was_revised}")

os.unlink(tmp_path)
print("\n✓ Example 3 complete. No GPU required.")
