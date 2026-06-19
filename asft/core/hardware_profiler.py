"""
Hardware Profiler — Auto-detect available compute resources and derive
optimal training configurations.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import psutil

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: Optional[str] = None
    is_cuda: bool = True

    @property
    def vram_used_gb(self) -> float:
        return self.vram_total_gb - self.vram_free_gb


@dataclass
class HardwareProfile:
    """
    Full hardware snapshot with derived training recommendations.
    All values are refreshed at creation time.
    """

    # ---- Raw hardware ----
    platform: str
    cpu_brand: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    ram_total_gb: float
    ram_available_gb: float
    storage_free_gb: float
    gpus: list[GPUInfo] = field(default_factory=list)
    has_cuda: bool = False
    has_mps: bool = False  # Apple Silicon

    # ---- Derived totals ----
    total_vram_gb: float = 0.0
    free_vram_gb: float = 0.0

    # ---- Recommendations ----
    recommended_precision: str = "fp32"        # fp32 / fp16 / bf16 / int8 / int4
    recommended_quantization: str = "none"     # none / 8bit / 4bit
    recommended_training_method: str = "lora"  # full / lora / qlora / sparse / asft
    recommended_batch_size: int = 1
    recommended_gradient_checkpointing: bool = False
    max_trainable_model_gb: float = 0.0
    offload_to_cpu: bool = False
    use_flash_attention: bool = False
    num_workers: int = 0

    def summary(self) -> str:
        lines = [
            f"Platform        : {self.platform}",
            f"CPU             : {self.cpu_brand} ({self.cpu_physical_cores}C/{self.cpu_logical_cores}T)",
            f"RAM             : {self.ram_total_gb:.1f} GB total, {self.ram_available_gb:.1f} GB free",
            f"Storage         : {self.storage_free_gb:.1f} GB free",
            f"CUDA            : {self.has_cuda}",
            f"MPS             : {self.has_mps}",
        ]
        for g in self.gpus:
            lines.append(
                f"GPU[{g.index}]         : {g.name} | VRAM {g.vram_total_gb:.1f} GB total, {g.vram_free_gb:.1f} GB free"
            )
        lines += [
            f"Precision       : {self.recommended_precision}",
            f"Quantization    : {self.recommended_quantization}",
            f"Training Method : {self.recommended_training_method}",
            f"Batch Size      : {self.recommended_batch_size}",
            f"Grad Checkpt    : {self.recommended_gradient_checkpointing}",
            f"CPU Offload     : {self.offload_to_cpu}",
            f"Flash Attn      : {self.use_flash_attention}",
            f"Max Model Size  : {self.max_trainable_model_gb:.1f} GB",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _get_cpu_brand() -> str:
    try:
        if platform.system() == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            brand, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return brand.strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def _detect_gpus() -> tuple[list[GPUInfo], bool, bool]:
    """Returns (gpus, has_cuda, has_mps)."""
    gpus: list[GPUInfo] = []
    has_cuda = False
    has_mps = False

    # Try PyTorch first (most reliable)
    try:
        import torch

        if torch.cuda.is_available():
            has_cuda = True
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                free = total - reserved
                cc = f"{props.major}.{props.minor}"
                gpus.append(
                    GPUInfo(
                        index=i,
                        name=props.name,
                        vram_total_gb=round(total, 2),
                        vram_free_gb=round(free, 2),
                        compute_capability=cc,
                        is_cuda=True,
                    )
                )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            has_mps = True
            # MPS shares system RAM; estimate available
            ram = psutil.virtual_memory()
            gpus.append(
                GPUInfo(
                    index=0,
                    name="Apple MPS",
                    vram_total_gb=round(ram.total / (1024**3) * 0.75, 2),
                    vram_free_gb=round(ram.available / (1024**3) * 0.75, 2),
                    compute_capability=None,
                    is_cuda=False,
                )
            )
    except ImportError:
        # Fallback: try nvidia-smi
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                timeout=5,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for i, line in enumerate(out.strip().splitlines()):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    name, total_mb, free_mb = parts
                    gpus.append(
                        GPUInfo(
                            index=i,
                            name=name,
                            vram_total_gb=round(int(total_mb) / 1024, 2),
                            vram_free_gb=round(int(free_mb) / 1024, 2),
                            is_cuda=True,
                        )
                    )
            has_cuda = bool(gpus)
        except Exception:
            pass

    return gpus, has_cuda, has_mps


def _check_flash_attention(has_cuda: bool) -> bool:
    if not has_cuda:
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------


def _derive_recommendations(profile: HardwareProfile) -> HardwareProfile:
    """
    Fill in all recommended_* fields based on the raw hardware data.

    Decision rules (in priority order):
      1. If GPU with ≥24 GB VRAM  → fp16/bf16, no quant, batch 4-8, full/lora
      2. If GPU with ≥12 GB VRAM  → bf16, 8-bit quant, batch 2-4, lora/qlora
      3. If GPU with  ≥6 GB VRAM  → fp16, 4-bit quant, batch 1-2, qlora/sparse
      4. If GPU with  <6 GB VRAM  → int4, 4-bit quant, batch 1, qlora, offload
      5. CPU-only ≥32 GB RAM      → int8, batch 1, sparse
      6. CPU-only  <32 GB RAM     → int4, batch 1, asft (minimal)
    """
    vram = profile.free_vram_gb
    ram = profile.ram_available_gb
    has_gpu = profile.has_cuda or profile.has_mps

    if has_gpu and vram >= 24:
        precision = "bf16"
        quant = "none"
        method = "lora"
        batch = 8
        gc = False
        offload = False
        max_model = vram * 0.85
    elif has_gpu and vram >= 12:
        precision = "bf16"
        quant = "8bit"
        method = "lora"
        batch = 4
        gc = True
        offload = False
        max_model = vram * 0.90
    elif has_gpu and vram >= 6:
        precision = "fp16"
        quant = "4bit"
        method = "qlora"
        batch = 2
        gc = True
        offload = False
        max_model = vram * 0.90
    elif has_gpu and vram >= 2:
        precision = "fp16"
        quant = "4bit"
        method = "qlora"
        batch = 1
        gc = True
        offload = True
        max_model = vram * 0.85 + ram * 0.3
    elif ram >= 32:
        precision = "int8"
        quant = "8bit"
        method = "sparse"
        batch = 1
        gc = True
        offload = False
        max_model = ram * 0.5
    else:
        precision = "int4"
        quant = "4bit"
        method = "asft"
        batch = 1
        gc = True
        offload = True
        max_model = ram * 0.4

    profile.recommended_precision = precision
    profile.recommended_quantization = quant
    profile.recommended_training_method = method
    profile.recommended_batch_size = batch
    profile.recommended_gradient_checkpointing = gc
    profile.offload_to_cpu = offload
    profile.max_trainable_model_gb = round(max_model, 2)
    profile.use_flash_attention = _check_flash_attention(profile.has_cuda)
    profile.num_workers = min(4, profile.cpu_physical_cores // 2)
    return profile


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_hardware() -> HardwareProfile:
    """
    Detect all available hardware and produce a complete HardwareProfile
    with derived training recommendations.
    """
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(".")
    gpus, has_cuda, has_mps = _detect_gpus()

    total_vram = sum(g.vram_total_gb for g in gpus)
    free_vram = sum(g.vram_free_gb for g in gpus)

    profile = HardwareProfile(
        platform=platform.system(),
        cpu_brand=_get_cpu_brand(),
        cpu_physical_cores=psutil.cpu_count(logical=False) or 1,
        cpu_logical_cores=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=round(mem.total / (1024**3), 2),
        ram_available_gb=round(mem.available / (1024**3), 2),
        storage_free_gb=round(disk.free / (1024**3), 2),
        gpus=gpus,
        has_cuda=has_cuda,
        has_mps=has_mps,
        total_vram_gb=round(total_vram, 2),
        free_vram_gb=round(free_vram, 2),
    )

    return _derive_recommendations(profile)


class HardwareProfiler:
    def __init__(self):
        self._profile = None

    def profile(self) -> HardwareProfile:
        self._profile = detect_hardware()
        return self._profile

    def get_profile(self) -> HardwareProfile:
        if self._profile is None:
            return self.profile()
        return self._profile

    def __str__(self):
        if self._profile:
            return self._profile.summary()
        return "HardwareProfiler(Uninitialized)"


if __name__ == "__main__":
    hw = detect_hardware()
    print(hw.summary())
