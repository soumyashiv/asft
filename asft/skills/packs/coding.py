"""Coding Skill Pack — Code generation, debugging, and review."""

from __future__ import annotations

import time

from asft.skills.skill_pack import SkillPack, SkillResult


class CodingSkillPack(SkillPack):
    def __init__(self, pack_dir=None):
        super().__init__("coding", pack_dir)
        self.meta.description = (
            "Code generation, debugging, and review across all programming languages"
        )
        self.meta.domain = "coding"
        self.meta.tags = ["code", "programming", "debug", "review", "algorithms"]

    def get_prompt_template(self) -> str:
        return (
            "You are an expert software engineer. Produce clean, efficient, well-commented code. "
            "Always include error handling and follow language best practices.\n\n"
            "Task: {task}\n\n"
            "Provide only the code solution with brief inline comments."
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
            metadata={"language": self._detect_language(task_input)},
        )

    def evaluate(self, sample_input: str, sample_output: str) -> float:
        # Heuristic: check for code blocks, no syntax indicators of errors
        has_code = "```" in sample_output or "def " in sample_output or "function " in sample_output
        has_error = any(w in sample_output.lower() for w in ["traceback", "error:", "exception:"])
        score = 0.7 if has_code else 0.3
        if has_error:
            score *= 0.5
        return min(1.0, score)

    def _run_model(self, prompt, model, tokenizer, kwargs):
        if model is None or tokenizer is None:
            return f"[CodingSkillPack] Would process: {prompt[:100]}..."
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _estimate_confidence(self, output: str) -> float:
        if not output or len(output) < 10:
            return 0.1
        has_code_block = "```" in output
        has_logic = any(kw in output for kw in ["def ", "class ", "for ", "if ", "return "])
        score = 0.5
        if has_code_block:
            score += 0.3
        if has_logic:
            score += 0.2
        return min(1.0, score)

    def _detect_language(self, text: str) -> str:
        text = text.lower()
        for lang in ["python", "javascript", "typescript", "java", "c++", "rust", "go", "sql"]:
            if lang in text:
                return lang
        return "unknown"
