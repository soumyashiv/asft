"""
LoRA / QLoRA Adapter — PEFT-based LoRA integration for ASFT.
Used as a hybrid with sparse training or as a standalone method.
Supports auto-detection of target modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Common target modules per architecture
_TARGET_MODULES_MAP = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "bert": ["query", "key", "value", "dense"],
    "default": ["q_proj", "v_proj"],
}


def _detect_target_modules(model) -> list[str]:
    """Auto-detect LoRA target modules based on model architecture name."""
    model_type = getattr(model.config, "model_type", "default").lower()
    for key, modules in _TARGET_MODULES_MAP.items():
        if key in model_type:
            return modules
    return _TARGET_MODULES_MAP["default"]


class LoRAAdapter:
    """
    Wraps PEFT LoRA / QLoRA around any HuggingFace model.
    Provides methods to: create, train, merge, save, and load adapters.
    """

    def __init__(self, config):  # LoRAConfig
        self._config = config
        self._model = None
        self._peft_model = None

    def wrap(self, model, quantization: str | None = None) -> object:
        """
        Wrap a base model with LoRA (or QLoRA if quantization is set).
        Returns the PEFT model.
        """
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            raise ImportError("Install peft: pip install peft")  # noqa: B904

        cfg = self._config
        target_modules = cfg.target_modules or _detect_target_modules(model)
        logger.info("LoRA target modules: %s", target_modules)

        # QLoRA: prepare for k-bit training if quantized
        if quantization in ("4bit", "8bit"):
            model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )

        self._model = model
        self._peft_model = get_peft_model(model, lora_cfg)
        self._peft_model.print_trainable_parameters()
        return self._peft_model

    def save(self, output_dir: str) -> None:
        """Save LoRA adapter weights to disk."""
        if self._peft_model is None:
            raise RuntimeError("No PEFT model initialized. Call wrap() first.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._peft_model.save_pretrained(output_dir)
        logger.info("LoRA adapter saved to %s", output_dir)

    def load(self, base_model, adapter_path: str) -> object:
        """Load a saved LoRA adapter onto a base model."""
        from peft import PeftModel

        self._peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        logger.info("LoRA adapter loaded from %s", adapter_path)
        return self._peft_model

    def merge_and_unload(self) -> object:
        """Merge LoRA weights into the base model and return a clean model."""
        if self._peft_model is None:
            raise RuntimeError("No PEFT model initialized.")
        merged = self._peft_model.merge_and_unload()
        logger.info("LoRA weights merged into base model")
        return merged

    def trainable_param_count(self) -> int:
        if self._peft_model is None:
            return 0
        return sum(p.numel() for p in self._peft_model.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        if self._peft_model is None:
            return 0
        return sum(p.numel() for p in self._peft_model.parameters())


def load_quantized_model(
    model_name: str,
    quantization: str = "4bit",
    cache_dir: str | None = None,
    trust_remote_code: bool = True,
):
    """
    Load a HuggingFace model with bitsandbytes quantization for QLoRA.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    logger.info("Model loaded: %s (quantization=%s)", model_name, quantization)
    return model
