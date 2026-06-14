"""Milestone 0 environment verification.

This script verifies local Python, package, CUDA, and NVIDIA driver visibility.
It does not load or download the Nemotron 3.5 ASR model.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable


MIN_PYTHON = (3, 10)

PACKAGE_SPECS = (
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("nemo", "nemo_toolkit"),
    ("huggingface_hub", "huggingface_hub"),
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    reason: str
    required: bool = True


def pass_check(name: str, reason: str, required: bool = True) -> CheckResult:
    return CheckResult(name=name, status="PASS", reason=reason, required=required)


def fail_check(name: str, reason: str, required: bool = True) -> CheckResult:
    return CheckResult(name=name, status="FAIL", reason=reason, required=required)


def python_version_string(version_info: tuple[int, int, int] | None = None) -> str:
    if version_info is None:
        version_info = sys.version_info[:3]
    return ".".join(str(part) for part in version_info)


def check_python_version(version_info: tuple[int, int, int] | None = None) -> CheckResult:
    if version_info is None:
        version_info = sys.version_info[:3]
    current = python_version_string(version_info)
    required = ".".join(str(part) for part in MIN_PYTHON)
    if version_info >= MIN_PYTHON:
        return pass_check("python_version", f"Python {current} satisfies >= {required}.")
    return fail_check(
        "python_version",
        f"Python {current} is too old. Install Python >= {required}.",
    )


def module_version(module: ModuleType, dist_name: str) -> str:
    version = getattr(module, "__version__", None)
    if version:
        return str(version)

    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def import_package(import_name: str, dist_name: str) -> tuple[CheckResult, str | None, ModuleType | None]:
    try:
        module = importlib.import_module(import_name)
    except ImportError as exc:
        return (
            fail_check(
                f"package:{import_name}",
                (
                    f"Could not import {import_name}: {exc}. "
                    "Install dependencies only after confirming CUDA/PyTorch/NeMo compatibility."
                ),
            ),
            None,
            None,
        )

    version = module_version(module, dist_name)
    if version == "unknown":
        reason = f"Imported {import_name}; version could not be determined."
    else:
        reason = f"Imported {import_name} {version}."
    return pass_check(f"package:{import_name}", reason), version, module


def check_packages(
    package_specs: Iterable[tuple[str, str]] = PACKAGE_SPECS,
) -> tuple[list[CheckResult], dict[str, str | None], dict[str, ModuleType]]:
    checks: list[CheckResult] = []
    package_versions: dict[str, str | None] = {}
    modules: dict[str, ModuleType] = {}

    for import_name, dist_name in package_specs:
        check, version, module = import_package(import_name, dist_name)
        checks.append(check)
        package_versions[import_name] = version
        if module is not None:
            modules[import_name] = module

    return checks, package_versions, modules


def bytes_to_gb(value: int | float) -> float:
    return round(float(value) / (1024**3), 2)


def bytes_to_mb(value: int | float) -> float:
    return round(float(value) / (1024**2), 2)


def check_cuda(torch_module: ModuleType | None) -> tuple[CheckResult, dict[str, Any]]:
    gpu_info = {
        "cuda_available": False,
        "gpu_name": None,
        "gpu_total_memory_gb": None,
        "gpu_memory_allocated_mb": None,
        "gpu_memory_reserved_mb": None,
    }

    if torch_module is None:
        return (
            fail_check(
                "cuda",
                "CUDA check skipped because torch could not be imported. Check PyTorch installation first.",
            ),
            gpu_info,
        )

    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 - defensive environment check
        return fail_check("cuda", f"torch.cuda.is_available() failed: {exc}."), gpu_info

    gpu_info["cuda_available"] = cuda_available
    if not cuda_available:
        return (
            fail_check(
                "cuda",
                "torch.cuda.is_available() returned False. Check NVIDIA driver, CUDA runtime, and PyTorch CUDA build.",
            ),
            gpu_info,
        )

    try:
        device_index = 0
        properties = torch_module.cuda.get_device_properties(device_index)
        gpu_info.update(
            {
                "gpu_name": torch_module.cuda.get_device_name(device_index),
                "gpu_total_memory_gb": bytes_to_gb(properties.total_memory),
                "gpu_memory_allocated_mb": bytes_to_mb(torch_module.cuda.memory_allocated(device_index)),
                "gpu_memory_reserved_mb": bytes_to_mb(torch_module.cuda.memory_reserved(device_index)),
            }
        )
    except Exception as exc:  # noqa: BLE001 - defensive environment check
        return (
            fail_check(
                "cuda_gpu_info",
                f"CUDA is available, but GPU details could not be read: {exc}.",
            ),
            gpu_info,
        )

    return pass_check("cuda", "CUDA is available and GPU memory metrics were read."), gpu_info


def run_nvidia_smi(
    run_func: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[CheckResult, bool, str]:
    try:
        result = run_func(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError:
        return (
            fail_check(
                "nvidia_smi",
                "nvidia-smi was not found on PATH. Check NVIDIA driver installation and PATH.",
            ),
            False,
            "",
        )
    except subprocess.TimeoutExpired:
        return fail_check("nvidia_smi", "nvidia-smi timed out after 30 seconds."), False, ""
    except Exception as exc:  # noqa: BLE001 - defensive environment check
        return fail_check("nvidia_smi", f"nvidia-smi failed to run: {exc}."), False, ""

    output = (result.stdout or "").strip()
    error_output = (result.stderr or "").strip()
    combined_output = output if output else error_output

    if result.returncode != 0:
        return (
            fail_check(
                "nvidia_smi",
                f"nvidia-smi exited with code {result.returncode}. Check NVIDIA driver/runtime setup.",
            ),
            False,
            combined_output,
        )

    return pass_check("nvidia_smi", "nvidia-smi completed successfully."), True, combined_output


def overall_status(checks: Iterable[CheckResult]) -> str:
    return "FAIL" if any(check.required and check.status != "PASS" for check in checks) else "PASS"


def build_report() -> tuple[dict[str, Any], str]:
    checks: list[CheckResult] = [check_python_version()]
    package_checks, package_versions, modules = check_packages()
    checks.extend(package_checks)

    cuda_check, gpu_info = check_cuda(modules.get("torch"))
    checks.append(cuda_check)

    nvidia_smi_check, nvidia_smi_available, nvidia_smi_output = run_nvidia_smi()
    checks.append(nvidia_smi_check)

    status = overall_status(checks)
    report: dict[str, Any] = {
        "python_version": python_version_string(),
        "package_versions": package_versions,
        "cuda_available": gpu_info["cuda_available"],
        "gpu_name": gpu_info["gpu_name"],
        "gpu_total_memory_gb": gpu_info["gpu_total_memory_gb"],
        "gpu_memory_allocated_mb": gpu_info["gpu_memory_allocated_mb"],
        "gpu_memory_reserved_mb": gpu_info["gpu_memory_reserved_mb"],
        "nvidia_smi_available": nvidia_smi_available,
        "checks": [asdict(check) for check in checks],
        "overall_status": status,
    }
    return report, nvidia_smi_output


def print_report(report: dict[str, Any], nvidia_smi_output: str) -> None:
    def display_value(value: Any) -> Any:
        return "not available" if value is None else value

    print("Milestone 0 Environment Verification")
    print("=" * 37)
    print(f"Python version: {report['python_version']}")
    print()

    print("Package versions:")
    for package_name, version in report["package_versions"].items():
        print(f"  {package_name}: {version or 'not available'}")
    print()

    print("CUDA:")
    print(f"  Available: {report['cuda_available']}")
    print(f"  GPU name: {display_value(report['gpu_name'])}")
    print(f"  Total memory (GB): {display_value(report['gpu_total_memory_gb'])}")
    print(f"  Allocated memory (MB): {display_value(report['gpu_memory_allocated_mb'])}")
    print(f"  Reserved memory (MB): {display_value(report['gpu_memory_reserved_mb'])}")
    print()

    print("nvidia-smi:")
    if nvidia_smi_output:
        print(nvidia_smi_output)
    else:
        print("  No nvidia-smi output captured.")
    print()

    print("Checks:")
    for check in report["checks"]:
        print(f"  [{check['status']}] {check['name']}: {check['reason']}")
    print()
    print(f"Overall status: {report['overall_status']}")


def write_json_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the Milestone 0 ASR environment.")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write a machine-readable environment report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report, nvidia_smi_output = build_report()
    print_report(report, nvidia_smi_output)

    if args.json_output:
        write_json_report(report, args.json_output)
        print()
        print(f"JSON report written to: {args.json_output}")

    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
