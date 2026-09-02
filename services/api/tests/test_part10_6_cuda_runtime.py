from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from services.api.app import cuda_runtime
from scripts.wakeword import gpu_stt_preflight


class WindowsCudaRuntimeTests(unittest.TestCase):
    def tearDown(self) -> None:
        cuda_runtime._DLL_DIRECTORY_HANDLES.clear()
        cuda_runtime._REGISTERED_DLL_DIRECTORIES.clear()
        cuda_runtime._PREPENDED_PATH_DIRECTORIES.clear()

    def test_non_windows_runtime_is_noop(self) -> None:
        with patch.object(cuda_runtime.sys, "platform", "linux"):
            self.assertEqual(cuda_runtime.candidate_windows_cuda_dll_directories(), [])
            self.assertEqual(cuda_runtime.configure_windows_cuda_dll_directories(), ())
            self.assertEqual(cuda_runtime.prepend_windows_cuda_process_path(), ())

    def test_mocking_os_name_on_ubuntu_cannot_construct_windows_path(self) -> None:
        with (
            patch.object(cuda_runtime.sys, "platform", "linux"),
            patch.object(cuda_runtime.os, "name", "nt"),
        ):
            self.assertEqual(cuda_runtime.candidate_windows_cuda_dll_directories(), [])

    def test_discovers_venv_nvidia_bin_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            cublas_bin = prefix / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
            cublas_bin.mkdir(parents=True)

            with (
                patch.object(cuda_runtime.sys, "platform", "win32"),
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
            patch.object(cuda_runtime.sys, "platform", "win32"),
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

    def test_process_path_prepend_is_scoped_and_idempotent(self) -> None:
        fake_dir = Path(r"C:\Bunnelby\nvidia\cublas\bin")
        original = r"C:\Windows\System32"
        with (
            patch.object(cuda_runtime.sys, "platform", "win32"),
            patch.object(cuda_runtime, "candidate_windows_cuda_dll_directories", return_value=[fake_dir]),
            patch.object(Path, "resolve", side_effect=lambda: fake_dir),
            patch.dict(cuda_runtime.os.environ, {"PATH": original}, clear=False),
        ):
            first = cuda_runtime.prepend_windows_cuda_process_path()
            second = cuda_runtime.prepend_windows_cuda_process_path()
            effective = cuda_runtime.os.environ["PATH"]

        self.assertEqual(first, (str(fake_dir),))
        self.assertEqual(second, ())
        self.assertTrue(effective.startswith(str(fake_dir) + os.pathsep))
        self.assertTrue(effective.endswith(original))


class GPUPreflightTests(unittest.TestCase):
    def test_preflight_consumes_lazy_segments_to_force_real_inference(self) -> None:
        consumed: list[bool] = []

        def segments():
            consumed.append(True)
            if False:
                yield None

        model = Mock()
        model.transcribe.return_value = (
            segments(),
            SimpleNamespace(language="en"),
        )
        fake_ctranslate2 = SimpleNamespace(
            __version__="test",
            get_cuda_device_count=lambda: 1,
            get_supported_compute_types=lambda *_args: {"int8_float16"},
        )
        configuration = cuda_runtime.WindowsCudaRuntimeConfiguration((), ())

        with (
            patch.object(gpu_stt_preflight, "configure_windows_cuda_runtime", return_value=configuration),
            patch.object(gpu_stt_preflight, "candidate_windows_cuda_dll_directories", return_value=[]),
            patch.object(gpu_stt_preflight, "_find_dll", return_value=Path("runtime.dll")),
            patch.object(gpu_stt_preflight, "_load_dll", return_value=(True, "loadable")),
            patch.dict("sys.modules", {"ctranslate2": fake_ctranslate2}),
            patch("faster_whisper.WhisperModel", return_value=model),
            patch.object(Path, "mkdir"),
        ):
            exit_code = gpu_stt_preflight.run(
                run_inference=True,
                compute_type="int8_float16",
                model_name="small",
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(consumed, [True])
        model.transcribe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
