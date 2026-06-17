"""Automation Skill Pack — Workflow scripting, pipelines, and system automation."""
from __future__ import annotations
import time
from asft.skills.skill_pack import SkillPack, SkillResult


class AutomationSkillPack(SkillPack):
    def __init__(self, pack_dir=None):
        super().__init__("automation", pack_dir)
        self.meta.description = "Workflow automation, scripting, pipeline design, and system orchestration"
        self.meta.domain = "automation"
        self.meta.tags = ["automation", "workflow", "script", "pipeline", "orchestration", "agent"]

    def get_prompt_template(self) -> str:
        return (
            "You are an automation and DevOps expert. Design efficient, reliable, "
            "and maintainable automation solutions. Include error handling, logging, "
            "and monitoring. Prefer idempotent operations.\n\n"
            "Automation Task: {task}\n\n"
            "Provide a complete automation solution with implementation details."
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
            metadata={"automation_type": self._detect_type(task_input)},
        )

    def evaluate(self, sample_input: str, sample_output: str) -> float:
        has_code = any(kw in sample_output for kw in ["def ", "import ", "#!/", "docker", "yaml"])
        has_steps = "\n-" in sample_output or "\n1." in sample_output
        has_error_handling = any(w in sample_output.lower() for w in ["error", "except", "try", "retry", "fallback"])
        return min(1.0, (0.4 if has_code else 0.1) + (0.2 if has_steps else 0) + (0.3 if has_error_handling else 0) + 0.1)

    def _run_model(self, prompt, model, tokenizer):
        if model is None or tokenizer is None:
            return f"[AutomationSkillPack] Would automate: {prompt[:80]}..."
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(**inputs, max_new_tokens=768, do_sample=False)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _estimate_confidence(self, output: str) -> float:
        if not output or len(output) < 20: return 0.1
        score = 0.3
        if "```" in output: score += 0.3
        if any(w in output.lower() for w in ["step", "workflow", "pipeline"]): score += 0.2
        if len(output) > 300: score += 0.2
        return min(1.0, score)

    def _detect_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ["docker", "kubernetes", "deploy"]): return "devops"
        if any(w in text_lower for w in ["cron", "schedule", "trigger"]): return "scheduling"
        if any(w in text_lower for w in ["api", "webhook", "http"]): return "integration"
        return "general"
