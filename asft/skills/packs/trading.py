"""Trading Skill Pack — Market analysis, signals, and financial reasoning."""

from __future__ import annotations

import time

from asft.skills.skill_pack import SkillPack, SkillResult


class TradingSkillPack(SkillPack):
    def __init__(self, pack_dir=None):
        super().__init__("trading", pack_dir)
        self.meta.description = (
            "Financial market analysis, trading signals, risk assessment, and portfolio management"
        )
        self.meta.domain = "trading"
        self.meta.tags = ["trading", "finance", "stocks", "risk", "portfolio", "market"]

    def get_prompt_template(self) -> str:
        return (
            "You are an expert financial analyst and quantitative trader. "
            "Provide objective, data-driven market analysis. "
            "Always include risk assessment. Do NOT provide financial advice — analysis only. "
            "Distinguish between analysis and speculation.\n\n"
            "Market Task: {task}\n\n"
            "Analysis:"
        )

    def process(self, task_input: str, model=None, tokenizer=None, **kwargs) -> SkillResult:
        start = time.time()
        prompt = self.get_prompt_template().format(task=task_input)
        output = self._run_model(prompt, model, tokenizer)
        duration = time.time() - start
        confidence = self._estimate_confidence(output)
        self.record_usage(success=True, score=confidence)
        return SkillResult(
            skill_name=self.meta.name,
            output=output,
            confidence=confidence,
            duration_seconds=round(duration, 3),
            metadata={"disclaimer": "Analysis only — not financial advice"},
        )

    def evaluate(self, sample_input: str, sample_output: str) -> float:
        has_analysis = any(
            w in sample_output.lower()
            for w in ["support", "resistance", "trend", "volume", "momentum", "risk"]
        )
        has_disclaimer = (
            "not financial advice" in sample_output.lower()
            or "analysis only" in sample_output.lower()
        )
        has_numbers = any(c.isdigit() for c in sample_output)
        return min(
            1.0,
            (0.4 if has_analysis else 0.1)
            + (0.3 if has_numbers else 0)
            + (0.3 if has_disclaimer else 0),
        )

    def _run_model(self, prompt, model, tokenizer):
        if model is None or tokenizer is None:
            return f"[TradingSkillPack] Analysis for: {prompt[:80]}..."
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.5)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _estimate_confidence(self, output: str) -> float:
        if not output or len(output) < 30:
            return 0.1
        score = 0.4
        if len(output) > 200:
            score += 0.2
        if any(w in output.lower() for w in ["risk", "trend", "signal"]):
            score += 0.2
        if any(c.isdigit() for c in output):
            score += 0.2
        return min(1.0, score)
