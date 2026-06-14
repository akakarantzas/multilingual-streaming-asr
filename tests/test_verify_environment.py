from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_environment.py"
SPEC = importlib.util.spec_from_file_location("verify_environment", SCRIPT_PATH)
verify_environment = importlib.util.module_from_spec(SPEC)
assert SPEC is not None
assert SPEC.loader is not None
sys.modules[SPEC.name] = verify_environment
SPEC.loader.exec_module(verify_environment)


class VerifyEnvironmentTests(unittest.TestCase):
    def test_python_version_check_passes_for_supported_version(self) -> None:
        check = verify_environment.check_python_version((3, 10, 0))

        self.assertEqual(check.status, "PASS")

    def test_python_version_check_fails_for_old_version(self) -> None:
        check = verify_environment.check_python_version((3, 9, 18))

        self.assertEqual(check.status, "FAIL")
        self.assertIn("too old", check.reason)

    def test_memory_conversion_helpers(self) -> None:
        self.assertEqual(verify_environment.bytes_to_gb(2 * 1024**3), 2.0)
        self.assertEqual(verify_environment.bytes_to_mb(512 * 1024**2), 512.0)

    def test_run_nvidia_smi_reports_missing_executable(self) -> None:
        def missing_runner(*_args, **_kwargs):
            raise FileNotFoundError

        check, available, output = verify_environment.run_nvidia_smi(missing_runner)

        self.assertEqual(check.status, "FAIL")
        self.assertFalse(available)
        self.assertEqual(output, "")
        self.assertIn("not found", check.reason)

    def test_run_nvidia_smi_reports_success(self) -> None:
        def successful_runner(*_args, **_kwargs):
            return subprocess.CompletedProcess(["nvidia-smi"], 0, stdout="driver ok\n", stderr="")

        check, available, output = verify_environment.run_nvidia_smi(successful_runner)

        self.assertEqual(check.status, "PASS")
        self.assertTrue(available)
        self.assertEqual(output, "driver ok")


if __name__ == "__main__":
    unittest.main()
