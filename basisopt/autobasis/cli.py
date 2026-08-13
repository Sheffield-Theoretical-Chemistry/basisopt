"""Colourful command-line interface for the auto-basis pipeline.

    python -m basisopt.autobasis run      my-run.yaml [--force]
    python -m basisopt.autobasis status   my-run.yaml
    python -m basisopt.autobasis validate my-run.yaml
    python -m basisopt.autobasis steps
    python -m basisopt.autobasis list-presets

Built on Typer + Rich, so ``--help`` and the status/validate tables come out
formatted and coloured. The commands are thin wrappers over ``pipeline`` /
``config`` -- the heavy imports happen inside command bodies so ``--help`` stays
instant and never spins up a backend.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
err_console = Console(stderr=True)

# One-line blurb per canonical step (drives `steps` and the run/status niceties).
# "M" marks the single molecular stage; the rest are atomic.
STEP_BLURBS = {
    "primitives": ("A", "grow primitive exponents from a Legendre expansion to the CBS target"),
    "reduction": ("A", "drop the least-important exponents while holding the target"),
    "contraction": ("A", "contract to natural atomic orbitals (NAOs)"),
    "uncontraction": ("M", "re-free the highest-contributing functions"),
    "purification": ("A", "purify the contraction coefficients (extended Davidson)"),
    "pruning": ("A", "zero the least-useful coefficients within an energy budget"),
    "polarisation": ("M", "grow d/f/g polarisation shells against reference molecules"),
}

APP_HELP = """[bold cyan]basisopt auto-basis[/] — build a basis set step by step, driven by energy requirements.

Each [bold]step[/] refines the basis and is resumable from [dim]workdir/manifest.json[/], so you can
re-run one step without redoing the rest.

[bold]Typical flow[/]
  [green]validate[/] a config → [green]run[/] it → check [green]status[/] → re-run a step with a narrower [cyan]steps:[/] list
"""


app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
    help=APP_HELP,
    epilog="Docs: [link]https://basisopt.readthedocs.io[/link]",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load(config: Path):
    """Load a pipeline config, turning a ConfigError into a red panel + exit 1."""
    from .config import ConfigError, load_config

    try:
        return load_config(config)
    except (ConfigError, FileNotFoundError) as exc:
        err_console.print(Panel(str(exc), title="[red]invalid config", border_style="red"))
        raise typer.Exit(1)


def _configure_logging(verbose: bool, quiet: bool) -> None:
    """Map ``--verbose``/``--quiet`` to a level and route ``bo_logger`` through Rich,
    so log lines share the CLI's visual system instead of clashing with colorlog."""
    import logging

    from basisopt.api import set_logger

    if verbose and quiet:
        raise typer.BadParameter("--verbose and --quiet are mutually exclusive")
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    set_logger(level, rich=True, console=err_console)


def _chain(steps) -> Text:
    """Render a step list as a coloured ``a → b → c`` chain."""
    text = Text()
    for i, step in enumerate(steps):
        if i:
            text.append("  →  ", style="dim")
        text.append(step, style="cyan")
    return text


def _header(cfg) -> Panel:
    """A compact identity banner for a config (name / element / tier / workdir)."""
    body = Text()
    body.append(f"{cfg.name}\n", style="bold white")
    body.append("element ", style="dim")
    body.append(f"{cfg.element}", style="bold magenta")
    if cfg.tier:
        body.append("   tier ", style="dim")
        body.append(f"{cfg.tier}", style="bold yellow")
    body.append("\nworkdir ", style="dim")
    body.append(str(cfg.workdir), style="cyan")
    return Panel(body, title="[bold cyan]auto-basis", border_style="cyan", expand=False)


def _status_table(cfg, manifest) -> Table:
    """A per-step table: which have completed, which are queued, which are skipped."""
    from .config import CANONICAL_STEPS

    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("", width=2)
    table.add_column("step", style="bold")
    table.add_column("state")
    table.add_column("backend", style="blue")
    table.add_column("when", style="dim")

    for step in CANONICAL_STEPS:
        if manifest.has(step):
            rec = manifest.step(step)
            table.add_row(
                Text("✓", style="bold green"),
                step,
                Text("done", style="green"),
                str(rec.get("backend", "")),
                str(rec.get("timestamp", "")),
            )
        elif step in cfg.steps:
            table.add_row(Text("▸", style="yellow"), step, Text("queued", style="yellow"), "", "")
        else:
            table.add_row(
                Text("·", style="dim"),
                Text(step, style="dim"),
                Text("skipped", style="dim"),
                "",
                "",
            )
    return table


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
@app.command()
def run(
    config: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to a pipeline YAML config"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-run steps even if already recorded in the manifest"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show DEBUG detail (per-iteration optimiser output)"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Only warnings/errors and step results"
    ),
) -> None:
    """[green]Run[/] the steps listed in a config file (resuming finished ones)."""
    from .config import CANONICAL_STEPS
    from .pipeline import run_pipeline

    _configure_logging(verbose, quiet)
    cfg = _load(config)
    console.print(_header(cfg))
    console.print("running ", Text("→ ", style="dim"), _chain(cfg.steps), "\n")

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        manifest = run_pipeline(str(config), force=force, timestamp=timestamp)
    except Exception as exc:  # surface backend/config failures clearly, non-zero exit
        err_console.print(
            Panel(f"{type(exc).__name__}: {exc}", title="[red]pipeline failed", border_style="red")
        )
        # Full traceback only when asked for; otherwise the one-line panel is enough.
        if verbose:
            err_console.print_exception(show_locals=False)
        raise typer.Exit(1)

    done = [s for s in CANONICAL_STEPS if manifest.has(s)]
    console.print()
    console.print(_status_table(cfg, manifest))
    console.print(
        Panel(
            f"[bold green]✓[/] {len(done)} step(s) complete  →  [cyan]{manifest.workdir}[/]",
            border_style="green",
            expand=False,
        )
    )


@app.command()
def status(
    config: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to a pipeline YAML config"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show DEBUG detail"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only warnings/errors"),
) -> None:
    """Show which steps have [green]completed[/] for a config, and which are queued."""
    from .manifest import Manifest

    _configure_logging(verbose, quiet)
    cfg = _load(config)
    manifest = Manifest.load_or_create(cfg.workdir, cfg.element)
    console.print(_header(cfg))
    console.print(_status_table(cfg, manifest))


@app.command()
def validate(
    config: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to a pipeline YAML config"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show DEBUG detail"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only warnings/errors"),
) -> None:
    """[green]Pre-flight[/] a config: parse it, summarise it, and check referenced files exist.

    Loads no backend and runs no calculation — safe to run anywhere, and answers
    "will this config do what I think, and are its paths real?" before a long run.
    """
    _configure_logging(verbose, quiet)
    cfg = _load(config)
    console.print(_header(cfg))

    summary = Table(box=None, pad_edge=False, show_header=False)
    summary.add_column(style="dim", justify="right")
    summary.add_column()
    summary.add_row("backend", Text(cfg.backend.default, style="blue"))
    summary.add_row("steps", _chain(cfg.steps))
    console.print(summary)

    issues: list[str] = []
    _check_file(getattr(cfg.reference, "geometry", None), "reference.geometry", issues)

    # Polarisation gets a dedicated molecule table -- the richest step surface.
    if "polarisation" in cfg.steps:
        console.print(_polarisation_panel(cfg, issues))

    console.print()
    if issues:
        body = "\n".join(f"[yellow]•[/] {m}" for m in issues)
        console.print(Panel(body, title="[yellow]warnings", border_style="yellow", expand=False))
        console.print(f"[green]config parses[/] · [yellow]{len(issues)} path warning(s)[/]")
    else:
        console.print(
            Panel(
                "[bold green]✓ config parses and all referenced files exist",
                border_style="green",
                expand=False,
            )
        )


@app.command("steps")
def steps_cmd() -> None:
    """List the seven canonical pipeline [cyan]steps[/] and what each one does."""
    from .config import CANONICAL_STEPS

    table = Table(title="[bold cyan]auto-basis steps", box=None, header_style="bold")
    table.add_column("#", style="dim", width=2)
    table.add_column("step", style="bold cyan")
    table.add_column("kind", justify="center", width=4)
    table.add_column("what it does")
    for i, step in enumerate(CANONICAL_STEPS, 1):
        kind, blurb = STEP_BLURBS[step]
        badge = Text("mol", style="magenta") if kind == "M" else Text("atom", style="green")
        table.add_row(str(i), step, badge, blurb)
    console.print(table)
    console.print(
        "[dim]atom[/] = optimised on the single atom · "
        "[dim]mol[/] = optimised on reference molecules"
    )


@app.command("list-presets")
def list_presets_cmd() -> None:
    """List the bundled tier [yellow]presets[/] (min / fast / mid / accu)."""
    from .config import list_presets

    presets = list_presets()
    if presets:
        table = Table(box=None, show_header=False)
        table.add_column(style="bold yellow")
        for name in presets:
            table.add_row(name)
        console.print(Panel(table, title="[bold]presets", border_style="yellow", expand=False))
    else:
        console.print(
            Panel(
                "No presets found.\nSet [cyan]BASISOPT_AUTOBASIS_PRESETS[/] to a directory of "
                "preset YAMLs,\nor use a path in a config's [cyan]extends:[/].",
                title="[yellow]no presets",
                border_style="yellow",
                expand=False,
            )
        )


# --------------------------------------------------------------------------- #
# validate helpers
# --------------------------------------------------------------------------- #
def _check_file(path, label: str, issues: list[str]) -> None:
    """Record a warning if a referenced path is set but missing."""
    if path and not Path(path).exists():
        issues.append(f"{label} not found: [cyan]{path}[/]")


def _polarisation_panel(cfg, issues: list[str]) -> Panel:
    """Summarise the polarisation step: mode/loss/target + a reference-molecule table."""
    pol = cfg.step_config("polarisation")
    head = Table(box=None, pad_edge=False, show_header=False)
    head.add_column(style="dim", justify="right")
    head.add_column()
    head.add_row("mode", Text(str(pol.get("mode", "greedy")), style="bold cyan"))
    head.add_row("loss", Text(str(pol.get("loss", "mean_per_electron")), style="bold"))
    if pol.get("target") is not None:
        head.add_row("target", f"{pol['target']} [dim](absolute)[/]")
    elif pol.get("target_ratio") is not None:
        head.add_row("target_ratio", f"{pol['target_ratio']} [dim](relative)[/]")
    if pol.get("spectator_basis"):
        head.add_row("spectator", str(pol["spectator_basis"]))

    _check_file(pol.get("input"), "polarisation.input", issues)
    if pol.get("input"):
        head.add_row("input", str(pol["input"]))

    mols = pol.get("molecules") or []
    mtable = Table(box=None, header_style="bold", pad_edge=False)
    mtable.add_column("", width=1)
    mtable.add_column("molecule", style="cyan")
    mtable.add_column("cbs_limit", justify="right")
    mtable.add_column("mult", justify="center")
    for m in mols:
        geom = m.get("geometry", "")
        exists = Path(geom).exists() if geom else False
        if geom and not exists:
            issues.append(f"molecule geometry not found: [cyan]{geom}[/]")
        mark = Text("✓", style="green") if exists else Text("✗", style="red")
        mtable.add_row(
            mark,
            Path(geom).name or "?",
            str(m.get("cbs_limit", "—")),
            str(m.get("multiplicity", "—")),
        )

    group = Table.grid(padding=(0, 0))
    group.add_row(head)
    if mols:
        group.add_row(Text(f"\n{len(mols)} reference molecule(s):", style="dim"))
        group.add_row(mtable)
    return Panel(group, title="[bold cyan]polarisation", border_style="cyan", expand=False)


# --------------------------------------------------------------------------- #
# entry point (kept as main(argv) so python -m basisopt.autobasis works)
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> int:
    """Run the Typer app, returning a process exit code (for ``__main__``)."""
    try:
        app(args=argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
