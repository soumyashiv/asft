"""
ASFT Example 1 — Hardware Detection & Skill Routing (no GPU required)
=======================================================================
Demonstrates: HardwareProfiler + SkillRouter + ConfidenceScorer
This example runs on any machine without a model or GPU.
"""

from asft.accuracy.confidence_scorer import ConfidenceScorer
from asft.core.config import ASFTConfig
from asft.core.hardware_profiler import detect_hardware
from asft.core.registry import registry
from asft.skills.skill_router import SkillRouter

print("=" * 60)
print("ASFT — Example 1: Hardware & Skill Routing")
print("=" * 60)

# 1. Detect hardware
print("\n[1] Detecting hardware...")
hw = detect_hardware()
print(hw.summary())

# 2. Init config
cfg = ASFTConfig()
cfg.apply_hardware_profile(hw)

# 3. Register skill packs
print("\n[2] Registering skill packs...")
from asft.skills.packs.automation import AutomationSkillPack
from asft.skills.packs.coding import CodingSkillPack
from asft.skills.packs.mathematics import MathematicsSkillPack
from asft.skills.packs.planning import PlanningSkillPack
from asft.skills.packs.research import ResearchSkillPack
from asft.skills.packs.trading import TradingSkillPack

packs = [
    CodingSkillPack(),
    ResearchSkillPack(),
    PlanningSkillPack(),
    MathematicsSkillPack(),
    TradingSkillPack(),
    AutomationSkillPack(),
]
for p in packs:
    registry.register_skill(p.meta.name, p)
print(f"  Registered: {registry.list('skill_packs')}")

# 4. Route various tasks
print("\n[3] Skill routing demo:")
router = SkillRouter(registry=registry)

tasks = [
    "Write a Python function to sort a list of dictionaries by a key",
    "Research the latest advances in large language model compression",
    "Create a 6-week project plan for building a mobile app",
    "Calculate the derivative of f(x) = 3x^3 + 2x^2 - 5x + 1",
    "Analyze the RSI indicator for a stock showing overbought conditions",
    "Automate the daily backup of a PostgreSQL database using cron",
]

for task in tasks:
    decision = router.route(task, strategy="single")
    best_skill = decision.selected_packs[0] if decision.selected_packs else "none"
    best_score = decision.scores.get(best_skill, 0.0)
    print(f"  [{best_skill:>15}] ({best_score:.2f}) — {task[:55]}...")

# 5. Confidence scoring demo
print("\n[4] Confidence scoring demo:")
scorer = ConfidenceScorer()
test_outputs = [
    "The answer is 42.",
    "I think maybe probably the result could be around 42, but I'm not sure.",
    "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "According to a recent study, experts agree this is widely known to be true.",
]
for output in test_outputs:
    score = scorer.score(output)
    print(f"  [{score.label:>6}] {score.composite:.3f} — {output[:60]}...")

# 6. Math skill direct computation
print("\n[5] Math direct computation:")
math_pack = MathematicsSkillPack()
for expr in ["2 + 2 * 5", "100 / 4 + 3", "(15 + 3) * 2"]:
    result = math_pack.process(expr)
    print(f"  {expr:<20} = {result.output}  (confidence={result.confidence:.2f})")

print("\n✓ Example 1 complete. No GPU required.")
