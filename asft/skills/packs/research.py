"""Research Skill Pack — Information synthesis, analysis, and literature review."""

from __future__ import annotations

import time

from asft.skills.skill_pack import SkillPack, SkillResult


class ResearchSkillPack(SkillPack):
    def __init__(self, pack_dir=None):
        super().__init__("research", pack_dir)
        self.meta.description = "Research, analysis, information synthesis, and literature review"
        self.meta.domain = "research"
        self.meta.tags = ["research", "analysis", "synthesis", "literature", "data"]

    def get_prompt_template(self) -> str:
        return (
            "You are a rigorous research analyst. Provide comprehensive, accurate, and "
            "well-sourced analysis. Structure your response clearly with sections. "
            "Distinguish between verified facts and estimates. Flag knowledge gaps.\n\n"
            "Research Task: {task}\n\n"
            "Provide a thorough, structured analysis."
        )

    def process(self, task_input: str, model=None, tokenizer=None, **kwargs) -> SkillResult:
        start = time.time()
        prompt = self.get_prompt_template().format(task=task_input)
        output = self._run_model(prompt, model, tokenizer, kwargs)
        duration = time.time() - start
        confidence = self._estimate_confidence(output)
        self.record_usage(success=True, score=confidence)
        return SkillResult(
            skill_name=self.meta.name,
            output=output,
            confidence=confidence,
            duration_seconds=round(duration, 3),
        )

    def evaluate(self, sample_input: str, sample_output: str) -> float:
        length_score = min(1.0, len(sample_output) / 500)
        has_structure = any(h in sample_output for h in ["##", "**", "1.", "-"])
        hedge_words = ["however", "additionally", "furthermore", "in contrast", "notably"]
        has_nuance = sum(1 for w in hedge_words if w in sample_output.lower())
        return min(
            1.0, length_score * 0.4 + (0.3 if has_structure else 0) + min(0.3, has_nuance * 0.1)
        )

    def _run_model(self, prompt, model, tokenizer, kwargs):
        if model is None or tokenizer is None:
            return f"[ResearchSkillPack] Would analyze: {prompt[:100]}..."
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(**inputs, max_new_tokens=768, do_sample=True, temperature=0.7)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _estimate_confidence(self, output: str) -> float:
        if not output or len(output) < 50:
            return 0.1
        length_score = min(0.5, len(output) / 1000)
        structure_bonus = 0.3 if any(c in output for c in ["##", "1.", "-"]) else 0.0
        return min(1.0, 0.2 + length_score + structure_bonus)
