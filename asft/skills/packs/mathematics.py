"""Mathematics Skill Pack — Computation, proofs, and symbolic reasoning."""
from __future__ import annotations

import re
import time

from asft.skills.skill_pack import SkillPack, SkillResult


class MathematicsSkillPack(SkillPack):
    def __init__(self, pack_dir=None):
        super().__init__("mathematics", pack_dir)
        self.meta.description = "Mathematical computation, symbolic reasoning, proofs, and statistics"
        self.meta.domain = "mathematics"
        self.meta.tags = ["math", "calculus", "algebra", "statistics", "proof", "computation"]

    def get_prompt_template(self) -> str:
        return (
            "You are a rigorous mathematician. Show all steps clearly. "
            "Verify your work. Use proper mathematical notation. "
            "If using approximations, state them explicitly.\n\n"
            "Math Problem: {task}\n\n"
            "Solution (show all steps):"
        )

    def process(self, task_input: str, model=None, tokenizer=None, **kwargs) -> SkillResult:
        start = time.time()
        # Try pure-Python computation first for simple expressions
        computed = self._try_direct_compute(task_input)
        if computed is not None:
            duration = time.time() - start
            self.record_usage(success=True, score=1.0)
            return SkillResult(
                skill_name=self.meta.name, output=str(computed),
                confidence=1.0, duration_seconds=round(duration, 4),
                metadata={"method": "direct_computation"},
            )

        prompt = self.get_prompt_template().format(task=task_input)
        output = self._run_model(prompt, model, tokenizer)
        duration = time.time() - start
        confidence = self._estimate_confidence(output)
        self.record_usage(success=True, score=confidence)
        return SkillResult(
            skill_name=self.meta.name, output=output,
            confidence=confidence, duration_seconds=round(duration, 3),
            metadata={"method": "model_inference"},
        )

    def evaluate(self, sample_input: str, sample_output: str) -> float:
        has_numbers = bool(re.search(r'\d', sample_output))
        has_steps = any(w in sample_output.lower() for w in ["therefore", "step", "=", "thus", "hence"])
        has_answer = any(w in sample_output.lower() for w in ["answer", "result", "=", "solution"])
        return min(1.0, (0.3 if has_numbers else 0) + (0.4 if has_steps else 0) + (0.3 if has_answer else 0))

    def _try_direct_compute(self, text: str):
        """Attempt safe eval of simple arithmetic expressions."""
        # Extract a clean math expression
        expr = re.sub(r'[^0-9+\-*/().^ ]', '', text).strip()
        if not expr or len(expr) < 2:
            return None
        try:
            # Replace ^ with ** for power
            expr = expr.replace('^', '**')
            result = eval(expr, {"__builtins__": {}})  # safe minimal eval
            return result
        except Exception:
            return None

    def _run_model(self, prompt, model, tokenizer):
        if model is None or tokenizer is None:
            return f"[MathematicsSkillPack] Would solve: {prompt[:80]}..."
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _estimate_confidence(self, output: str) -> float:
        if not output: return 0.1
        has_eq = "=" in output
        has_num = bool(re.search(r'\d+\.?\d*', output))
        return min(1.0, 0.3 + (0.4 if has_eq else 0) + (0.3 if has_num else 0))
