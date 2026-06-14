from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate benchmark plots from JSON artifacts.")
    parser.add_argument("--input-dir", required=True, help="Directory containing benchmark JSON files.")
    parser.add_argument(
        "--output-dir",
        default="reports/figures",
        help="Directory for generated PNG plots.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        artifacts = load_benchmark_artifacts(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generated = generate_plots(artifacts, output_dir)
        if generated:
            print("Generated plots:")
            for path in generated:
                print(f"- {path}")
        else:
            print("No plots generated; benchmark artifacts did not contain plottable data.")
    except ValueError as exc:
        print(f"Plot generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


def load_benchmark_artifacts(input_dir: str) -> dict:
    directory = Path(input_dir)
    if not directory.exists():
        raise ValueError(f"Input directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Input path is not a directory: {directory}")

    artifacts: dict[str, object] = {}
    for filename in (
        "profile_summary.json",
        "concurrency_summary.json",
        "gpu_snapshots.json",
    ):
        path = directory / filename
        if path.exists():
            artifacts[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return artifacts


def generate_plots(artifacts: dict, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    generated: list[Path] = []
    generated.extend(_plot_latency_distribution(artifacts, output_dir, plt))
    generated.extend(_plot_rtf_vs_chunk_size(artifacts, output_dir, plt))
    generated.extend(_plot_gpu_util_over_time(artifacts, output_dir, plt))
    generated.extend(_plot_concurrency_latency(artifacts, output_dir, plt))
    return generated


def has_latency_distribution_data(artifacts: dict) -> bool:
    return bool(artifacts.get("profile_summary", {}).get("per_input_latencies_ms"))


def has_rtf_chunk_data(artifacts: dict) -> bool:
    profile = artifacts.get("profile_summary", {})
    return bool(profile.get("chunk_metrics"))


def has_gpu_util_time_series(artifacts: dict) -> bool:
    snapshots = artifacts.get("gpu_snapshots", [])
    return any(item.get("gpu_utilization_pct") is not None for item in snapshots)


def has_concurrency_latency_data(artifacts: dict) -> bool:
    return bool(artifacts.get("concurrency_summary"))


def _plot_latency_distribution(artifacts: dict, output_dir: Path, plt) -> list[Path]:
    if not has_latency_distribution_data(artifacts):
        print("Skipping latency_distribution.png: missing per-input latency data.")
        return []
    latencies = artifacts["profile_summary"]["per_input_latencies_ms"]
    path = output_dir / "latency_distribution.png"
    plt.figure()
    plt.hist(latencies, bins=min(20, max(1, len(latencies))))
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("Per-file Latency Distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return [path]


def _plot_rtf_vs_chunk_size(artifacts: dict, output_dir: Path, plt) -> list[Path]:
    if not has_rtf_chunk_data(artifacts):
        print("Skipping rtf_vs_chunk_size.png: missing chunk-size RTF data.")
        return []
    rows = artifacts["profile_summary"]["chunk_metrics"]
    path = output_dir / "rtf_vs_chunk_size.png"
    plt.figure()
    plt.plot([row["chunk_ms"] for row in rows], [row["rtf"] for row in rows], marker="o")
    plt.xlabel("Chunk size (ms)")
    plt.ylabel("RTF")
    plt.title("RTF vs Chunk Size")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return [path]


def _plot_gpu_util_over_time(artifacts: dict, output_dir: Path, plt) -> list[Path]:
    if not has_gpu_util_time_series(artifacts):
        print("Skipping gpu_util_over_time.png: missing GPU utilization time series.")
        return []
    snapshots = [
        item for item in artifacts["gpu_snapshots"] if item.get("gpu_utilization_pct") is not None
    ]
    path = output_dir / "gpu_util_over_time.png"
    plt.figure()
    plt.plot(range(len(snapshots)), [item["gpu_utilization_pct"] for item in snapshots], marker="o")
    plt.xlabel("Snapshot index")
    plt.ylabel("GPU utilization (%)")
    plt.title("GPU Utilization Over Benchmark Run")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return [path]


def _plot_concurrency_latency(artifacts: dict, output_dir: Path, plt) -> list[Path]:
    if not has_concurrency_latency_data(artifacts):
        print("Skipping concurrency_latency.png: missing concurrency summary data.")
        return []
    rows = artifacts["concurrency_summary"]
    path = output_dir / "concurrency_latency.png"
    plt.figure()
    plt.plot([row["stream_count"] for row in rows], [row["p95_latency_ms"] for row in rows], marker="o")
    plt.xlabel("Concurrent streams")
    plt.ylabel("P95 latency (ms)")
    plt.title("P95 Latency vs Concurrent Streams")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return [path]


if __name__ == "__main__":
    raise SystemExit(main())
