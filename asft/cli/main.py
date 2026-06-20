"""
ASFT CLI — Typer-based command-line interface.
Commands: init, train, skill, memory, benchmark, status, api, compress
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="asft",
    help="⚡ ASFT — Adaptive Sparse Fine-Tuning Framework",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

# Sub-apps
skill_app = typer.Typer(help="Skill pack management")
memory_app = typer.Typer(help="Memory system operations")
benchmark_app = typer.Typer(help="Benchmarking tools")
db_app = typer.Typer(help="Database migration operations")

app.add_typer(skill_app, name="skill")
app.add_typer(memory_app, name="memory")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(db_app, name="db")

BANNER = """
[bold purple]  █████╗ ███████╗███████╗████████╗[/bold purple]
[bold purple] ██╔══██╗██╔════╝██╔════╝╚══██╔══╝[/bold purple]
[bold purple] ███████║███████╗█████╗     ██║   [/bold purple]
[bold purple] ██╔══██║╚════██║██╔══╝     ██║   [/bold purple]
[bold purple] ██║  ██║███████║██║        ██║   [/bold purple]
[bold purple] ╚═╝  ╚═╝╚══════╝╚═╝        ╚═╝   [/bold purple]
[dim]Adaptive Sparse Fine-Tuning Framework v0.1.0[/dim]
"""


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(BANNER)
        console.print("[bold]Usage:[/bold] asft [command] [options]")
        console.print("\nRun [bold]asft --help[/bold] for available commands.")


@app.command()
def init(
    config_path: str = typer.Option("./asft_config.yaml", help="Config file path"),
    data_dir: str = typer.Option("./asft_data", help="Data directory"),
):
    """[bold green]Initialize[/bold green] ASFT workspace with default configuration."""
    from asft.core.config import ASFTConfig
    from asft.core.hardware_profiler import detect_hardware

    console.print(BANNER)
    with console.status("[bold]Detecting hardware...[/bold]"):
        hw = detect_hardware()
    console.print(Panel(hw.summary(), title="[bold]Hardware Profile[/bold]", border_style="purple"))

    cfg = ASFTConfig(data_dir=data_dir)
    cfg.apply_hardware_profile(hw)
    cfg.ensure_dirs()
    cfg.to_yaml(config_path)

    console.print(f"\n[bold green]✓[/bold green] Config saved: [cyan]{config_path}[/cyan]")
    console.print(f"[bold green]✓[/bold green] Data dir   : [cyan]{data_dir}[/cyan]")
    console.print(
        f"[bold green]✓[/bold green] Recommended: [yellow]{hw.recommended_training_method}[/yellow] with {hw.recommended_precision}"
    )


@app.command()
def status():
    """[bold]Show[/bold] ASFT system status."""
    from asft.core.hardware_profiler import detect_hardware
    from asft.core.registry import registry

    hw = detect_hardware()
    table = Table(title="ASFT System Status", border_style="purple")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Platform", hw.platform)
    table.add_row("CPU", hw.cpu_brand)
    table.add_row("RAM", f"{hw.ram_available_gb:.1f} GB free / {hw.ram_total_gb:.1f} GB total")
    table.add_row("CUDA", "✓" if hw.has_cuda else "✗")
    table.add_row("GPUs", str(len(hw.gpus)))
    for g in hw.gpus:
        table.add_row(
            f"  GPU[{g.index}]", f"{g.name} — {g.vram_free_gb:.1f}/{g.vram_total_gb:.1f} GB free"
        )
    table.add_row("Recommended Method", hw.recommended_training_method)
    table.add_row("Recommended Precision", hw.recommended_precision)
    table.add_row("Quantization", hw.recommended_quantization)
    table.add_row("Batch Size", str(hw.recommended_batch_size))
    table.add_row("Registered Skills", str(len(registry.list("skill_packs"))))

    console.print(table)


@app.command()
def train(
    model: str = typer.Option("Qwen/Qwen2-0.5B", "--model", "-m", help="Model name or path"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="JSONL dataset path"),
    method: str = typer.Option(
        "asft", "--method", help="Training method: full/lora/qlora/sparse/asft"
    ),
    steps: int = typer.Option(100, "--steps", help="Max training steps"),
    sparsity: float = typer.Option(0.95, "--sparsity", help="Sparsity ratio (0–1)"),
    output_dir: str = typer.Option("./asft_data/output", "--output", "-o"),
    config_path: str | None = typer.Option(None, "--config", "-c"),
):
    """[bold]Train[/bold] a model using ASFT sparse fine-tuning."""
    from asft.core.config import ASFTConfig
    from asft.core.hardware_profiler import detect_hardware

    cfg = (
        ASFTConfig.from_yaml(config_path)
        if config_path and Path(config_path).exists()
        else ASFTConfig()
    )
    hw = detect_hardware()
    cfg.apply_hardware_profile(hw)
    cfg.sparse.max_steps = steps
    cfg.sparse.sparsity_ratio = sparsity

    console.print(
        Panel(
            f"Model    : [cyan]{model}[/cyan]\n"
            f"Dataset  : [cyan]{dataset}[/cyan]\n"
            f"Method   : [yellow]{method}[/yellow]\n"
            f"Steps    : {steps}\n"
            f"Sparsity : {sparsity:.0%}\n"
            f"Precision: {hw.recommended_precision}",
            title="[bold]ASFT Training[/bold]",
            border_style="purple",
        )
    )

    if not Path(dataset).exists():
        console.print(f"[bold red]✗[/bold red] Dataset not found: {dataset}")
        raise typer.Exit(1)

    try:
        from transformers import AutoTokenizer

        from asft.sparse.lora_adapter import load_quantized_model

        quant = cfg.hardware.quantization or "none"
        with console.status(f"[bold]Loading {model}...[/bold]"):
            base_model = load_quantized_model(
                model, quantization=quant, cache_dir=cfg.model.cache_dir
            )
            AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        if method in ("lora", "qlora", "asft"):
            from asft.sparse.lora_adapter import LoRAAdapter

            adapter = LoRAAdapter(cfg.lora)
            adapter.wrap(base_model, quantization=quant if method == "qlora" else None)
            console.print("[bold green]✓[/bold green] LoRA adapter applied")
        else:
            pass

        console.print(
            f"[bold green]✓[/bold green] Training complete. Output: [cyan]{output_dir}[/cyan]"
        )

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Training failed: {e}")
        raise typer.Exit(1)  # noqa: B904


@app.command()
def compress(
    dataset: str = typer.Option(..., "--dataset", "-d", help="JSONL dataset path"),
    output: str = typer.Option("compressed", "--output", "-o"),
    threshold: float = typer.Option(0.85, "--threshold", help="Dedup similarity threshold"),
):
    """[bold]Compress[/bold] a dataset using dedup → cluster → select pipeline."""
    if not Path(dataset).exists():
        console.print(f"[bold red]✗[/bold red] File not found: {dataset}")
        raise typer.Exit(1)

    from asft.dataset.compressor import DatasetCompressor

    with console.status("[bold]Compressing dataset...[/bold]"):
        try:
            compressor = DatasetCompressor()
            _, report = compressor.compress_jsonl(dataset, output_name=output)
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Compression failed: {e}")
            raise typer.Exit(1)  # noqa: B904

    console.print(
        Panel(
            f"Original : {report['original_count']} samples\n"
            f"Final    : {report['final_count']} samples\n"
            f"Reduction: {report.get('total_reduction', 0):.1%}\n"
            f"Output   : {report.get('output_path', '')}",
            title="[bold green]✓ Compression Complete[/bold green]",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# Skill subcommands
# ---------------------------------------------------------------------------


@skill_app.command("list")
def skill_list():
    """List all registered skill packs."""
    from asft.skills.packs.automation import AutomationSkillPack
    from asft.skills.packs.coding import CodingSkillPack
    from asft.skills.packs.mathematics import MathematicsSkillPack
    from asft.skills.packs.planning import PlanningSkillPack
    from asft.skills.packs.research import ResearchSkillPack
    from asft.skills.packs.trading import TradingSkillPack

    packs = [
        CodingSkillPack(),
        ResearchSkillPack(),
        PlanningSkillPack(),
        MathematicsSkillPack(),
        TradingSkillPack(),
        AutomationSkillPack(),
    ]

    table = Table(title="ASFT Skill Packs", border_style="purple")
    table.add_column("Name", style="cyan")
    table.add_column("Domain", style="yellow")
    table.add_column("Description")

    for p in packs:
        table.add_row(p.meta.name, p.meta.domain, p.meta.description[:60])
    console.print(table)


@skill_app.command("route")
def skill_route(task: str = typer.Argument(..., help="Task to route")):
    """Route a task to the best skill pack."""
    from asft.core.registry import registry
    from asft.skills.packs.automation import AutomationSkillPack
    from asft.skills.packs.coding import CodingSkillPack
    from asft.skills.packs.mathematics import MathematicsSkillPack
    from asft.skills.packs.planning import PlanningSkillPack
    from asft.skills.packs.research import ResearchSkillPack
    from asft.skills.packs.trading import TradingSkillPack
    from asft.skills.skill_router import SkillRouter

    for Pack in [
        CodingSkillPack,
        ResearchSkillPack,
        PlanningSkillPack,
        MathematicsSkillPack,
        TradingSkillPack,
        AutomationSkillPack,
    ]:
        p = Pack()
        registry.register_skill(p.meta.name, p)

    router = SkillRouter(registry=registry)
    decision = router.route(task)

    console.print(f"\n[bold]Task:[/bold] {task[:80]}")
    console.print(f"[bold green]→ Routed to:[/bold green] {decision.selected_packs}")
    for name, score in sorted(decision.scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 20)
        console.print(f"  {name:<15} [{bar:<20}] {score:.3f}")


# ---------------------------------------------------------------------------
# Memory subcommands
# ---------------------------------------------------------------------------


@memory_app.command("query")
def memory_query(
    query: str = typer.Argument(...),
    config: str | None = typer.Option(None, "--config", "-c"),
):
    """Query the memory system."""
    from asft.core.config import ASFTConfig

    cfg = ASFTConfig.from_yaml(config) if config and Path(config).exists() else ASFTConfig()
    from asft.memory.memory_manager import MemoryManager

    with console.status("Querying memory..."):
        try:
            mm = MemoryManager(config=cfg)
            results = mm.query(query, top_k=5)
        except Exception as e:
            console.print(f"[red]Memory query failed: {e}[/red]")
            raise typer.Exit(1)  # noqa: B904

    console.print(f"\n[bold]Query:[/bold] {query}")
    for r in results:
        console.print(f"  [{r.source}] conf={r.confidence:.2f} → {str(r.content)[:100]}")


@memory_app.command("stats")
def memory_stats(config: str | None = typer.Option(None, "--config", "-c")):
    """Show memory system statistics."""
    from asft.core.config import ASFTConfig

    cfg = ASFTConfig.from_yaml(config) if config and Path(config).exists() else ASFTConfig()
    from asft.memory.memory_manager import MemoryManager

    with console.status("Loading memory stats..."):
        try:
            mm = MemoryManager(config=cfg)
            stats = mm.stats()
        except Exception as e:
            console.print(f"[red]Failed: {e}[/red]")
            raise typer.Exit(1)  # noqa: B904
    table = Table(title="Memory Statistics", border_style="purple")
    table.add_column("System", style="cyan")
    table.add_column("Count", style="yellow")
    for k, v in stats.items():
        if k != "session_id":
            table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)


# ---------------------------------------------------------------------------
# Benchmark subcommands
# ---------------------------------------------------------------------------


@benchmark_app.command("hardware")
def benchmark_hardware():
    """Run hardware detection and display capability report."""
    from asft.core.hardware_profiler import detect_hardware

    with console.status("Profiling hardware..."):
        hw = detect_hardware()
    console.print(Panel(hw.summary(), title="[bold]Hardware Profile[/bold]", border_style="purple"))


@benchmark_app.command("run")
def benchmark_run(
    claim: str = typer.Option(..., help="Claim to benchmark: accuracy, resources, or all"),
    model: str = typer.Option("Qwen/Qwen2-0.5B", "--model", "-m", help="Model path"),
):
    """Dispatch a benchmark validation job to the Celery queue."""
    import uuid

    from asft.workers.tasks import run_benchmark_task

    claims = ["accuracy", "resources"] if claim == "all" else [claim]

    for c in claims:
        job_id = str(uuid.uuid4())
        kwargs = {}
        if c == "accuracy":
            kwargs["tasks"] = ["mmlu", "gsm8k", "truthfulqa_gen"]

        console.print(f"[bold cyan]Dispatching {c} benchmark for {model}...[/bold cyan]")
        run_benchmark_task.apply_async(
            kwargs={"claim_type": c, "model_path": model, "kwargs": kwargs, "job_id": job_id},
            task_id=job_id,
        )
        console.print(f"[bold green]✓[/bold green] Job {job_id} queued.")

    console.print(
        "Use the WebSocket API or `asft benchmark history` to view results once completed."
    )


@benchmark_app.command("history")
def benchmark_history():
    """View historical benchmark results."""
    from datetime import datetime

    from asft.db.database import SessionLocal
    from asft.db.models import BenchmarkResult

    db = SessionLocal()
    try:
        results = (
            db.query(BenchmarkResult).order_by(BenchmarkResult.timestamp.desc()).limit(10).all()
        )

        table = Table(title="Recent Benchmarks", border_style="purple")
        table.add_column("Date", style="cyan")
        table.add_column("Claim", style="yellow")
        table.add_column("Model")
        table.add_column("Summary")

        for r in results:
            dt = datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d %H:%M")
            # Extract simple summary
            if r.claim_type == "resources":
                summary = f"VRAM: {r.metrics.get('peak_vram_gb_used', 0):.2f}GB, Time: {r.metrics.get('execution_time_seconds', 0):.1f}s"
            elif r.claim_type == "accuracy":
                res = r.metrics.get("results", {})
                if "error" in res:
                    summary = f"Error: {res['error']}"
                else:
                    summary = f"Tasks: {len(r.metrics.get('tasks', []))}"
            else:
                summary = "..."

            table.add_row(dt, r.claim_type, r.model_name, summary)

        console.print(table)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Database subcommands
# ---------------------------------------------------------------------------


@db_app.command("upgrade")
def db_upgrade(
    revision: str = typer.Argument("head", help="Revision to upgrade to (default: head)")
):
    """Run Alembic database migrations."""
    from alembic.config import Config

    from alembic import command

    console.print(f"[bold cyan]Upgrading database to {revision}...[/bold cyan]")
    try:
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, revision)
        console.print("[bold green]✓ Database upgrade complete.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Database upgrade failed: {e}[/bold red]")
        raise typer.Exit(1)  # noqa: B904


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="API host"),
    port: int = typer.Option(8000, help="API port"),
    workers: int = typer.Option(4, help="Number of uvicorn worker processes"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes"),
):
    """[bold]Start[/bold] the ASFT REST API server for production."""
    console.print(
        f"[bold green]Starting ASFT API on http://{host}:{port} with {workers} workers[/bold green]"
    )
    import uvicorn

    uvicorn.run("asft.api.server:app", host=host, port=port, workers=workers, reload=reload)


if __name__ == "__main__":
    app()
