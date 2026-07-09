"""Command-line interface for the auto-basis pipeline.

    python -m basisopt.autobasis run    my-run.yaml [--force]
    python -m basisopt.autobasis status my-run.yaml
    python -m basisopt.autobasis list-presets
"""

from __future__ import annotations

import argparse
import datetime
from typing import Optional

from .config import CANONICAL_STEPS, list_presets, load_config
from .manifest import Manifest
from .pipeline import run_pipeline


def _cmd_run(args) -> int:
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    manifest = run_pipeline(args.config, force=args.force, timestamp=timestamp)
    done = ", ".join(s for s in CANONICAL_STEPS if manifest.has(s)) or "(none)"
    print(f"Completed steps in {manifest.workdir}: {done}")
    return 0


def _cmd_status(args) -> int:
    cfg = load_config(args.config)
    manifest = Manifest.load_or_create(cfg.workdir, cfg.element)
    print(f"Run '{cfg.name}'  element={cfg.element}  tier={cfg.tier}  workdir={cfg.workdir}")
    print(f"Steps requested this config: {', '.join(cfg.steps)}")
    for step in CANONICAL_STEPS:
        if manifest.has(step):
            rec = manifest.step(step)
            print(f"  [x] {step:<14} {rec.get('backend', ''):<7} {rec.get('timestamp', '')}")
        else:
            marker = ">" if step in cfg.steps else " "
            print(f"  [{marker}] {step:<14} (not run)")
    return 0


def _cmd_list_presets(args) -> int:
    presets = list_presets()
    if presets:
        print("Presets:", ", ".join(presets))
    else:
        print("No presets found. Set BASISOPT_AUTOBASIS_PRESETS to a directory "
              "of preset YAMLs, or use a path in 'extends:'.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="basisopt.autobasis", description="Run the auto-basis generation pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the steps listed in a config file")
    p_run.add_argument("config", help="path to a pipeline YAML config")
    p_run.add_argument("--force", action="store_true", help="rerun steps even if already done")
    p_run.set_defaults(func=_cmd_run)

    p_status = sub.add_parser("status", help="show which steps have completed for a config")
    p_status.add_argument("config", help="path to a pipeline YAML config")
    p_status.set_defaults(func=_cmd_status)

    p_list = sub.add_parser("list-presets", help="list bundled tier presets")
    p_list.set_defaults(func=_cmd_list_presets)

    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
