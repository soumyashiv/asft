"""
ASFT Knowledge Distillation Engine — Transfer capability from large to small models.

WHAT IS KNOWLEDGE DISTILLATION:
    Distillation (Hinton et al. 2015) trains a small "student" model to mimic
    the output DISTRIBUTION of a large "teacher" model, rather than just the
    hard class labels.

    The key insight: a teacher trained on "cat" vs "dog" might output:
        [cat: 0.90, dog: 0.08, tiger: 0.02]
    This distribution reveals that "cat" is somewhat like "tiger" and unlike "dog".
    This information is richer than just the label "cat" and helps the student
    learn faster with fewer examples.

DISTILLATION LOSS (Hinton et al. 2015, Eq. 4):
    L = α · CE(σ(z_s), y_hard)           ← hard label loss (standard CE)
      + (1-α) · T² · KL(σ(z_t/T), σ(z_s/T))  ← soft target loss

    Where:
      z_s   = student logits
      z_t   = teacher logits
      T     = temperature (softens the distribution; T>1 reveals more structure)
      α     = mix weight (0.5 = equal; 1.0 = hard labels only; 0.0 = soft only)
      T²    = scale factor to maintain gradient magnitude (standard practice)

TEMPERATURE INTUITION:
    - T=1: Standard CE. Teacher's argmax dominates.
    - T=4: Softer distribution. Reveals similarities between classes.
    - T=10: Very soft. Useful when teacher is very confident everywhere.
    Recommended range: T ∈ [2, 8] (Hinton 2015 used T=8 for speech models)

VALIDATED RESULTS:
    - DistilBERT (Sanh et al. 2019): 66% of BERT size, 97% of BERT performance
      on GLUE, 60% faster inference.
    - TinyBERT (Jiao et al. 2020): 7.5x smaller, 9.4x faster, 96.8% of BERT.
    - DistilGPT-2 (Wolf et al. 2019): 82M vs 124M params, near-GPT-2 perplexity.

REALISTIC EXPECTATIONS (for LLM distillation):
    - A student 50% of the teacher's size: expect 80–95% of teacher performance
    - A student 20% of the teacher's size: expect 70–85% of teacher performance
    - Factors that improve results: more distillation data, intermediate-layer
      matching (hidden state distillation, attention distillation — not implemented here)
    - Factors that degrade results: very different architectures (teacher/student
      must be same family for feature distillation; any model for logit distillation)

WHEN IT FAILS:
    - Student model too small for the task complexity (capacity bottleneck)
    - Insufficient distillation dataset (fewer than 10k samples is risky)
    - Teacher and student have very different tokenizers (tokenizer mismatch
      makes logit alignment impossible — use response distillation instead)
    - Tasks requiring genuinely long chains of reasoning (CoT distillation
      requires specialized techniques beyond logit matching)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for a knowledge distillation run."""
    teacher_model_name: str
    student_model_name: str
    output_dir: str = "./asft_data/distilled"
    dataset_path: str = ""
    text_field: str = "text"

    # Hinton hyperparameters
    temperature: float = 4.0        # T: higher = softer teacher distribution
    alpha: float = 0.5              # mix: 0=soft only, 1=hard only, 0.5=equal

    # Training
    max_steps: int = 500
    learning_rate: float = 5e-5
    batch_size: int = 2
    max_seq_len: int = 512
    gradient_accumulation_steps: int = 4
    quantize_teacher: bool = True   # Load teacher in 4-bit to save memory

    # Validation
    eval_steps: int = 100
    save_steps: int = 500


@dataclass
class DistillationResult:
    """Result of a distillation run."""
    status: str                          # "completed" | "failed"
    student_model_name: str
    output_dir: str
    total_steps: int = 0
    final_distill_loss: float = 0.0
    final_student_loss: float = 0.0
    training_time_seconds: float = 0.0
    teacher_param_billions: float = 0.0
    student_param_billions: float = 0.0
    compression_ratio: float = 0.0       # teacher/student param count
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class KnowledgeDistiller:
    """
    Distills knowledge from a large teacher model into a smaller student model.

    Supports:
        - Logit distillation (this file): teacher → student via soft targets
        - Future: hidden-state distillation, attention distillation (v2)

    Usage:
        config = DistillationConfig(
            teacher_model_name="Qwen/Qwen2-7B",
            student_model_name="Qwen/Qwen2-1.5B",
            temperature=4.0,
            alpha=0.5,
        )
        distiller = KnowledgeDistiller()
        result = distiller.distill(config)
    """

    def distill(self, config: DistillationConfig) -> DistillationResult:
        """
        Run the full distillation pipeline.

        Steps:
            1. Load teacher (frozen, optionally 4-bit quantized)
            2. Load student (trainable)
            3. Load dataset
            4. Run distillation training loop
            5. Save student model
        """
        start_time = time.time()
        warnings = []
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "KnowledgeDistiller: teacher=%s student=%s T=%.1f α=%.2f",
            config.teacher_model_name, config.student_model_name,
            config.temperature, config.alpha
        )

        try:
            teacher, teacher_tok, student, student_tok, t_params, s_params = \
                self._load_models(config, warnings)

            dataloader = self._load_dataset(config, student_tok)
            if dataloader is None:
                return DistillationResult(
                    status="failed",
                    student_model_name=config.student_model_name,
                    output_dir=str(out_dir),
                    error_message="Failed to load dataset",
                    warnings=warnings,
                )

            result = self._training_loop(
                config, teacher, student, student_tok, dataloader, out_dir
            )

            elapsed = time.time() - start_time
            result.training_time_seconds = elapsed
            result.teacher_param_billions = t_params
            result.student_param_billions = s_params
            result.compression_ratio = t_params / max(0.001, s_params)
            result.warnings = warnings

            logger.info(
                "Distillation completed | steps=%d loss=%.4f time=%.0fs "
                "compression=%.1fx",
                result.total_steps, result.final_distill_loss,
                elapsed, result.compression_ratio
            )
            return result

        except Exception as e:
            logger.exception("Distillation failed")
            return DistillationResult(
                status="failed",
                student_model_name=config.student_model_name,
                output_dir=str(out_dir),
                training_time_seconds=time.time() - start_time,
                error_message=str(e),
                warnings=warnings,
            )

    # ------------------------------------------------------------------
    # Private implementation
    # ------------------------------------------------------------------

    def _load_models(self, config: DistillationConfig, warnings: List[str]):
        """Load teacher (frozen) and student (trainable)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Teacher: load frozen, optionally quantized
        logger.info("Loading teacher: %s", config.teacher_model_name)
        teacher_tok = AutoTokenizer.from_pretrained(
            config.teacher_model_name, use_fast=True
        )
        if teacher_tok.pad_token is None:
            teacher_tok.pad_token = teacher_tok.eos_token

        teacher_kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": config.teacher_model_name,
            "trust_remote_code": False,
            "torch_dtype": torch.float16,
        }
        if config.quantize_teacher and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            teacher_kwargs["quantization_config"] = bnb_config
            teacher_kwargs["device_map"] = "auto"
        elif torch.cuda.is_available():
            teacher_kwargs["device_map"] = "auto"

        teacher = AutoModelForCausalLM.from_pretrained(**teacher_kwargs)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)  # Teacher is always frozen

        # Student: load trainable
        logger.info("Loading student: %s", config.student_model_name)
        student_tok = AutoTokenizer.from_pretrained(
            config.student_model_name, use_fast=True
        )
        if student_tok.pad_token is None:
            student_tok.pad_token = student_tok.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        student = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=False,
        ).to(device)

        # Check tokenizer compatibility
        if teacher_tok.vocab_size != student_tok.vocab_size:
            warnings.append(
                f"Teacher and student have different vocabulary sizes "
                f"({teacher_tok.vocab_size} vs {student_tok.vocab_size}). "
                "Logit distillation requires matching vocabularies. "
                "Consider using response distillation (SFT on teacher outputs) instead."
            )

        t_params = sum(p.numel() for p in teacher.parameters()) / 1e9
        s_params = sum(p.numel() for p in student.parameters()) / 1e9
        logger.info("Teacher: %.2fB params | Student: %.2fB params", t_params, s_params)

        return teacher, teacher_tok, student, student_tok, t_params, s_params

    def _load_dataset(self, config: DistillationConfig, tokenizer):
        """Load and tokenize the distillation dataset."""
        if not config.dataset_path:
            logger.warning("No dataset path provided — generating dummy data")
            return self._dummy_dataloader(tokenizer, config)
        try:
            from datasets import load_dataset
            import torch
            from torch.utils.data import DataLoader

            ext = Path(config.dataset_path).suffix
            if ext in (".jsonl", ".json"):
                dataset = load_dataset("json", data_files=config.dataset_path, split="train")
            else:
                dataset = load_dataset(config.dataset_path, split="train")

            def tokenize(batch):
                return tokenizer(
                    batch[config.text_field],
                    truncation=True,
                    max_length=config.max_seq_len,
                    padding="max_length",
                    return_tensors="pt",
                )

            dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
            dataset.set_format("torch")
            return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        except Exception as e:
            logger.error("Dataset loading failed: %s", e)
            return None

    def _dummy_dataloader(self, tokenizer, config: DistillationConfig):
        """Generate synthetic data for testing distillation pipeline."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        dummy_text = "The quick brown fox jumps over the lazy dog. " * 20
        enc = tokenizer(
            dummy_text, return_tensors="pt", truncation=True,
            max_length=config.max_seq_len, padding="max_length"
        )
        dataset = TensorDataset(
            enc["input_ids"].repeat(64, 1),
            enc["attention_mask"].repeat(64, 1),
        )
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    def _training_loop(
        self, config: DistillationConfig, teacher, student, tokenizer,
        dataloader, out_dir: Path
    ) -> DistillationResult:
        """
        Run the Hinton distillation training loop.

        Loss = α · CE(student_logits, hard_labels)
             + (1-α) · T² · KL(softmax(teacher_logits/T) || softmax(student_logits/T))
        """
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW

        device = next(student.parameters()).device
        optimizer = AdamW(student.parameters(), lr=config.learning_rate)

        step = 0
        running_distill_loss = 0.0
        running_student_loss = 0.0

        student.train()

        while step < config.max_steps:
            for batch in dataloader:
                if step >= config.max_steps:
                    break

                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(device)
                    attention_mask = batch[1].to(device) if len(batch) > 1 else None
                else:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)

                labels = input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100

                # Teacher forward (no gradient)
                with torch.no_grad():
                    teacher_device = next(teacher.parameters()).device
                    t_out = teacher(
                        input_ids=input_ids.to(teacher_device),
                        attention_mask=attention_mask.to(teacher_device) if attention_mask is not None else None,
                    )
                    t_logits = t_out.logits.to(device)

                # Student forward
                s_out = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                s_logits = s_out.logits

                # Handle vocabulary size mismatch (truncate teacher to student vocab size)
                min_vocab = min(t_logits.shape[-1], s_logits.shape[-1])
                t_logits_trimmed = t_logits[..., :min_vocab]
                s_logits_trimmed = s_logits[..., :min_vocab]

                # Hard label loss (standard cross-entropy, already computed by HF)
                hard_loss = s_out.loss if s_out.loss is not None else torch.tensor(0.0, device=device)

                # Soft target loss (Hinton KL divergence)
                T = config.temperature
                soft_teacher = F.softmax(t_logits_trimmed / T, dim=-1)
                soft_student = F.log_softmax(s_logits_trimmed / T, dim=-1)
                # KL(p_teacher || p_student) = Σ p_t * (log p_t - log p_s)
                soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)

                # Combined loss
                loss = config.alpha * hard_loss + (1 - config.alpha) * soft_loss

                # Backward
                if step % config.gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                loss.backward()

                if (step + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()

                running_distill_loss += float(soft_loss)
                running_student_loss += float(hard_loss)
                step += 1

                if step % 50 == 0:
                    avg_distill = running_distill_loss / max(1, step)
                    avg_student = running_student_loss / max(1, step)
                    logger.info(
                        "Distill step %d/%d | soft_loss=%.4f | hard_loss=%.4f",
                        step, config.max_steps, avg_distill, avg_student
                    )

        # Save student
        student.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        logger.info("Student model saved to %s", out_dir)

        return DistillationResult(
            status="completed",
            student_model_name=config.student_model_name,
            output_dir=str(out_dir),
            total_steps=step,
            final_distill_loss=running_distill_loss / max(1, step),
            final_student_loss=running_student_loss / max(1, step),
        )
