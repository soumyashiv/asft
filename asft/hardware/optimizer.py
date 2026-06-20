"""
Hardware-Adaptive Optimization — Quantizer, Offloader, and Batch Scheduler.
Auto-selects quantization levels, offloading strategies, and batch sizes
based on the current hardware profile. No manual tuning required.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ===========================================================================
# Quantizer
# ===========================================================================

class Quantizer:
    """
    Auto-selects and applies quantization to HuggingFace models.
    Supports: fp32, fp16, bf16, int8 (bitsandbytes), int4 (bitsandbytes nf4).
    """

    def load_quantized(
        self,
        model_name: str,
        precision: str = "bf16",
        quantization: str = "none",
        cache_dir: str | None = None,
        trust_remote_code: bool = True,
        device_map: str = "auto",
    ):
        """Load a model with the specified precision and quantization."""
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = None
        dtype = torch.float32

        if precision == "bf16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif precision in ("fp16", "f16"):
            dtype = torch.float16

        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            dtype = None
            logger.info("Loading %s with 4-bit NF4 quantization", model_name)
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            dtype = None
            logger.info("Loading %s with 8-bit quantization", model_name)
        else:
            logger.info("Loading %s with precision=%s", model_name, precision)

        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if bnb_config:
            kwargs["quantization_config"] = bnb_config
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        logger.info("Model loaded: %s", model_name)
        return model

    def estimate_memory_gb(self, n_params_billions: float, precision: str,
                            quantization: str) -> float:
        """Estimate peak GPU memory for a model."""
        base_gb = n_params_billions * 2  # fp16 baseline
        if quantization == "4bit":
            return base_gb * 0.3
        if quantization == "8bit":
            return base_gb * 0.55
        if precision == "fp32":
            return base_gb * 2.0
        if precision in ("bf16", "fp16"):
            return base_gb
        return base_gb


# ===========================================================================
# CPU Offloader
# ===========================================================================

class Offloader:
    """
    Manages CPU offloading strategy for models that exceed GPU memory.
    Uses HuggingFace Accelerate for transparent device placement.
    """

    def apply_cpu_offload(self, model, max_gpu_memory_gb: float | None = None) -> Any:
        """Apply CPU offloading to a model."""
        try:
            import torch
            from accelerate import dispatch_model, infer_auto_device_map

            if max_gpu_memory_gb:
                memory_map = {"cpu": "30GiB"}
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        memory_map[i] = f"{max_gpu_memory_gb:.0f}GiB"
                device_map = infer_auto_device_map(model, max_memory=memory_map)
            else:
                device_map = "auto"

            model = dispatch_model(model, device_map=device_map)
            logger.info("CPU offloading applied")
            return model
        except ImportError:
            logger.warning("accelerate not installed — CPU offloading unavailable")
            return model
        except Exception as e:
            logger.warning("CPU offloading failed: %s", e)
            return model


# ===========================================================================
# Batch Scheduler
# ===========================================================================

class BatchScheduler:
    """
    Dynamically selects optimal batch size based on available VRAM/RAM.
    Includes OOM recovery with automatic batch size reduction.
    """

    def __init__(self, initial_batch_size: int = 1, min_batch_size: int = 1,
                 max_batch_size: int = 64):
        self._batch_size = initial_batch_size
        self._min = min_batch_size
        self._max = max_batch_size
        self._oom_count = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def recommend_from_hardware(self, profile) -> int:
        """Set batch size from hardware profile recommendations."""
        self._batch_size = min(
            self._max,
            max(self._min, profile.recommended_batch_size)
        )
        return self._batch_size

    def on_oom(self) -> int:
        """Called when an OOM error occurs. Halves batch size."""
        self._oom_count += 1
        self._batch_size = max(self._min, self._batch_size // 2)
        logger.warning("OOM #%d — batch size reduced to %d", self._oom_count, self._batch_size)
        return self._batch_size

    def on_success(self) -> None:
        """Track successful batch processing."""
        pass  # Could implement adaptive growth here

    def safe_run(self, fn, *args, **kwargs) -> Any:
        """Run fn with automatic OOM recovery."""
        import torch
        while self._batch_size >= self._min:
            try:
                return fn(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    new_bs = self.on_oom()
                    if new_bs < self._min:
                        raise RuntimeError("Batch size cannot be reduced further — OOM")
                    logger.info("Retrying with batch_size=%d", new_bs)
                else:
                    raise
        raise RuntimeError("Could not find a viable batch size")
