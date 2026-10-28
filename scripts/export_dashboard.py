#!/usr/bin/env python3
"""Export an approved experiment dashboard as a lightweight static site."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def export_dashboard(source: Path, output: Path, overwrite: bool = False) -> list[Path]:
    """Copy only the experiment dashboard HTML files into a Pages-ready layout.

    The global dashboard is renamed to ``index.html``. Per-run dashboards keep
    the relative ``runs/<run_id>/dashboard.html`` paths expected by its panels.
    Existing output files are never removed; with ``overwrite=True``, only
    matching HTML files in this export are replaced.
    """
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    global_dashboard = source / "dashboard.html"
    if not source.is_dir():
        raise ValueError(f"source directory does not exist: {source}")
    if not global_dashboard.is_file():
        raise ValueError(f"global dashboard not found: {global_dashboard}")

    run_dashboards = sorted((source / "runs").glob("*/dashboard.html"))
    if not run_dashboards:
        raise ValueError(f"no per-run dashboards found under {source / 'runs'}")

    copies = [(global_dashboard, output / "index.html")]
    copies.extend(
        (dashboard, output / "runs" / dashboard.parent.name / "dashboard.html")
        for dashboard in run_dashboards
    )
    conflicts = [destination for _, destination in copies if destination.exists()]
    if conflicts and not overwrite:
        shown = ", ".join(str(path) for path in conflicts[:5])
        remainder = len(conflicts) - 5
        if remainder > 0:
            shown += f", and {remainder} more"
        raise FileExistsError(
            "export would replace existing files; pass --overwrite to replace "
            f"only these matching HTML files: {shown}"
        )

    for _, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
    for origin, destination in copies:
        shutil.copyfile(origin, destination)
    return [destination for _, destination in copies]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export an approved experiment's global and per-run dashboards "
            "to a static-site directory. No source files are changed or removed."
        )
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only same-named dashboard HTML files in the export",
    )
    args = parser.parse_args()
    try:
        exported = export_dashboard(args.source, args.output, args.overwrite)
    except (FileExistsError, ValueError, OSError) as error:
        print(f"Dashboard export failed: {error}", file=sys.stderr)
        return 2

    print(f"Exported {len(exported)} HTML files to {args.output.expanduser().resolve()}")
    print("The source experiment directory was left unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
