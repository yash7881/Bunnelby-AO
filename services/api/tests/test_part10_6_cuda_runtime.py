from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from services.api.app import cuda_runtime


class WindowsCudaRuntimeTests(unittest.TestCase):
    def tearDown(self) -> None:
        cuda_runtime._DLL_DIRECTORY_HANDLES.clear()
        cuda_runtime._REGISTERED_DLL_DIRECTORIES.clear()

    def test_non_windows_runtime_is_noop(self) -> None:
        with patch.object(cuda_runtime.os, "name", "posix"):
            self.assertEqual(cuda_runtime.candidate_windows_cuda_dll_directories(), [])
            self.assertEqual(cuda_runtime.configure_windows_cuda_dll_directories(), ())

    def test_discovers_venv_nvidia_bin_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            cublas_bin = prefix / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
            cublas_bin.mkdir(parents=True)

            with (
                patch.object(cuda_runtime.os, "name", "nt"),
                patch.object(cuda_runtime.sys, "prefix", str(prefix)),
                patch.dict(cuda_runtime.os.environ, {"ProgramFiles": str(prefix / "Program Files")}),
            ):
                found = cuda_runtime.candidate_windows_cuda_dll_directories()

        self.assertIn(cublas_bin, found)

    def test_registration_is_process_local_and_idempotent(self) -> None:
        fake_dir = Path(r"C:\Bunnelby\nvidia\cublas\bin")
        handle = object()
        add_dll_directory = Mock(return_value=handle)

        with (
            patch.object(cuda_runtime.os, "name", "nt"),
            patch.object(cuda_runtime.os, "add_dll_directory", add_dll_directory, create=True),
            patch.object(cuda_runtime, "candidate_windows_cuda_dll_directories", return_value=[fake_dir]),
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "resolve", side_effect=lambda: fake_dir),
        ):
            first = cuda_runtime.configure_windows_cuda_dll_directories()
            second = cuda_runtime.configure_windows_cuda_dll_directories()

        self.assertEqual(first, (str(fake_dir),))
        self.assertEqual(second, ())
        add_dll_directory.assert_called_once_with(str(fake_dir))
        self.assertEqual(cuda_runtime._DLL_DIRECTORY_HANDLES, [handle])


if __name__ == "__main__":
    unittest.main()
