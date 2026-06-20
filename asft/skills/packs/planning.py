"""Planning Skill Pack — Project planning, task decomposition, and strategy."""
from __future__ import annotations

import time

from asft.skills.skill_pack import SkillPack, SkillResult


class PlanningSkillPack(SkillPack):
    def __init__(self, pack_dir=None):
        super().__init__("planning", pack_dir)
        self.meta.description = "Strategic planning, task decomposition, roadmaps, and project management"
        self.meta.domain = "planning"
        self.meta.tags = ["planning", "strategy", "project", "roadmap", "tasks"]

    def get_prompt_template(self) -> str:
        return (
            "You are a strategic planning expert. Break down complex goals into clear, "
            "actionable steps with realistic timelines. Consider dependencies, risks, "
            "and resource constraints.\n\n"
            "Planning Task: {task}\n\n"
            "Provide a structured plan with phases, milestones, and success criteria."
        )

    def process(self, task_input: str, model=None, tokenizer=None, **kwargs) -> SkillResult:
        start = time.time()
        prompt = self.get_prompt_template().format(task=task_input)
        output = self._run_model(prompt, model, tokenizer)
        duration = time.time() - start
        confidence = self._estimate_confidence(output)
        self.record_usage(success=True, score=confidence)
        return SkillResult(
            skill_name=self.meta.name, output=output,
            confidence=confidence, duration_seconds=round(duration, 3),
            metadata={"steps_detected": output.count("\n- ") + output.count("\n1.")},
        )

    def evaluate(self, sample_input: str, sample_output: str) -> float:
        has_phases = any(w in sample_output.lower() for w in ["phase", "step", "milestone", "stage"])
        has_list = "\n-" in sample_output or "\n1." in sample_output
        has_timeline = any(w in sample_output.lower() for w in ["week", "day", "month", "hour"])
        score = (0.4 if has_phases else 0.0) + (0.3 if has_list else 0.0) + (0.3 if has_timeline else 0.0)
        return min(1.0, score + 0.1)

    def _run_model(self, prompt, model, tokenizer):
        if model is None or tokenizer is None:
            return f"[PlanningSkillPack] Plan for: {prompt[:80]}..."
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(**inputs, max_new_tokens=600, do_sample=True, temperature=0.6)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _estimate_confidence(self, output: str) -> float:
        score = 0.3
        if len(output) > 200: score += 0.2
        if "\n" in output: score += 0.2
        if any(w in output.lower() for w in ["step", "phase", "milestone"]): score += 0.3
        return min(1.0, score)
