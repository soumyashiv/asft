"""
ASFT Example 2 — Memory Systems (no GPU required)
===================================================
Demonstrates: WorkingMemory + EpisodicMemory + SemanticMemory + MemoryManager
All memory operations use SQLite — no model needed.
"""
import time
from asft.core.config import ASFTConfig
from asft.memory.working_memory import WorkingMemory
from asft.memory.episodic_memory import EpisodicMemory
from asft.memory.semantic_memory import SemanticMemory

print("=" * 60)
print("ASFT — Example 2: Memory Systems")
print("=" * 60)

# 1. Working Memory
print("\n[1] Working Memory")
wm = WorkingMemory(max_items=10, default_ttl_seconds=3600)
wm.set("current_task", "Analyze Q3 financial report", ttl=60)
wm.set("user_name", "Alex", tags=["user"])
wm.set("context", {"mode": "research", "depth": "deep"}, tags=["context"])

print(f"  Items stored: {wm.count()}")
print(f"  current_task: {wm.get('current_task')}")
print(f"  Tagged 'user': {[e.key for e in wm.search_by_tag('user')]}")
wm.set("temp", "expires_soon", ttl=0.001)
time.sleep(0.01)
wm.prune_expired()
print(f"  After TTL prune: {wm.count()}")

# 2. Episodic Memory
print("\n[2] Episodic Memory (SQLite)")
em = EpisodicMemory(db_path=":memory:")
em.record("task_start", {"task": "analyze_report", "domain": "finance"}, success=True, duration=0.5)
em.record("task_complete", {"task": "analyze_report", "output_len": 450}, success=True, duration=2.1)
em.record("task_error", {"task": "code_gen", "error": "syntax_error"}, success=False, duration=0.3)
em.record("task_complete", {"task": "math_solve"}, success=True, duration=0.8)
em.record("task_complete", {"task": "research"}, success=True, duration=3.2)

print(f"  Total events: {em.count()}")
print(f"  Failure rate: {em.failure_rate():.1%}")
print(f"  Recent events: {[e['event_type'] for e in em.query(limit=3)]}")

# 3. Semantic Memory
print("\n[3] Semantic Memory (SQLite)")
sm = SemanticMemory(db_path=":memory:")
sm.store_fact("Python", "is_a", "programming language", source="textbook", confidence=1.0)
sm.store_fact("ASFT", "is_a", "fine-tuning framework", source="docs", confidence=1.0)
sm.store_fact("Qwen2", "is_a", "language model", source="paper", confidence=0.99)
sm.store_fact("ASFT", "uses", "sparse training", source="spec", confidence=1.0)
sm.store_fact("LoRA", "is_a", "parameter efficient method", source="paper", confidence=1.0)

print(f"  Facts stored: {sm.count_facts()}")
q = sm.query_by_subject("ASFT")
for fact in q:
    print(f"  ASFT {fact.predicate} {fact.obj} (conf={fact.confidence})")

# Check: Is Python a programming language?
result = sm.check("Python", "is_a", "programming language")
print(f"  Python is a programming language? {result}")

# 4. Memory Manager quick test
print("\n[4] Memory Manager Integration")
cfg = ASFTConfig(data_dir=":memory:")
try:
    from asft.memory.memory_manager import MemoryManager
    mm = MemoryManager(config=cfg)
    mm.learn_fact("Transformers", "is_a", "library", source="docs")
    mm.record_task("code_gen", True, duration=1.5, context={"lang": "python"})

    results = mm.query("Transformers library", top_k=3)
    print(f"  Query results: {len(results)}")
    for r in results:
        print(f"  [{r.source}] conf={r.confidence:.2f}: {str(r.content)[:60]}")

    stats = mm.stats()
    print(f"  Memory stats: {stats}")
except Exception as e:
    print(f"  Note: {e} (requires non-in-memory config for full test)")

print("\n✓ Example 2 complete. All memory ops use SQLite — no GPU needed.")
