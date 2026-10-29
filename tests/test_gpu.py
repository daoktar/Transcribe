"""Tests for transcribe.gpu (VRAM probe and the process-wide GPU lock)."""
import contextlib
import threading
from types import SimpleNamespace
from unittest.mock import patch

from transcribe import gpu

# conftest patches gpu.cuda_total_gb for every test; keep the real one for these.
_real_cuda_total_gb = gpu.cuda_total_gb


def _torch(cuda, total_gb=3.6):
    props = SimpleNamespace(total_memory=int(total_gb * 2**30))
    return SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: cuda, get_device_properties=lambda _i: props,
    ))


class TestCudaTotalGb:
    def setup_method(self):
        _real_cuda_total_gb.cache_clear()

    def teardown_method(self):
        _real_cuda_total_gb.cache_clear()

    def _linux_with_driver(self):
        return (patch("transcribe.gpu.sys.platform", "linux"),
                patch("transcribe.gpu.os.path.exists", return_value=True))

    def test_reports_gib(self):
        plat, drv = self._linux_with_driver()
        with plat, drv, \
             patch("transcribe.gpu.importlib.import_module", return_value=_torch(True, 3.6)):
            assert abs(_real_cuda_total_gb() - 3.6) < 1e-6

    def test_macos_never_imports_torch(self):
        with patch("transcribe.gpu.sys.platform", "darwin"), \
             patch("transcribe.gpu.importlib.import_module") as imp:
            assert _real_cuda_total_gb() is None
        imp.assert_not_called()

    def test_wsl2_gpu_bridge_counts_as_cuda(self):
        # WSL2: no /proc/driver/nvidia, CUDA comes through /dev/dxg from the Windows driver.
        with patch("transcribe.gpu.sys.platform", "linux"), \
             patch("transcribe.gpu.os.path.exists", side_effect=lambda p: p == "/dev/dxg"), \
             patch("transcribe.gpu.importlib.import_module", return_value=_torch(True, 3.6)):
            assert abs(_real_cuda_total_gb() - 3.6) < 1e-6

    def test_linux_without_nvidia_driver_never_imports_torch(self):
        with patch("transcribe.gpu.sys.platform", "linux"), \
             patch("transcribe.gpu.os.path.exists", return_value=False), \
             patch("transcribe.gpu.importlib.import_module") as imp:
            assert _real_cuda_total_gb() is None
        imp.assert_not_called()

    def test_no_cuda(self):
        plat, drv = self._linux_with_driver()
        with plat, drv, \
             patch("transcribe.gpu.importlib.import_module", return_value=_torch(False)):
            assert _real_cuda_total_gb() is None

    def test_no_torch(self):
        plat, drv = self._linux_with_driver()
        with plat, drv, \
             patch("transcribe.gpu.importlib.import_module", side_effect=ImportError):
            assert _real_cuda_total_gb() is None


class TestExclusive:
    def test_noop_without_cuda(self):
        with patch("transcribe.gpu.cuda_total_gb", return_value=None):
            assert isinstance(gpu.exclusive(), contextlib.nullcontext)

    def test_lock_with_cuda(self):
        with patch("transcribe.gpu.cuda_total_gb", return_value=3.6):
            assert gpu.exclusive() is gpu._LOCK

    def test_decorated_calls_do_not_overlap(self):
        inside, overlap = [0], []
        gate = threading.Event()

        @gpu.exclusive_call
        def job():
            inside[0] += 1
            overlap.append(inside[0])
            gate.wait(0.05)
            inside[0] -= 1

        with patch("transcribe.gpu.cuda_total_gb", return_value=3.6):
            threads = [threading.Thread(target=job) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert overlap == [1, 1, 1, 1]

    def test_reentrant(self):
        @gpu.exclusive_call
        def outer():
            return inner()

        @gpu.exclusive_call
        def inner():
            return "ok"

        with patch("transcribe.gpu.cuda_total_gb", return_value=3.6):
            assert outer() == "ok"


class TestFailedStageFreesItsModels:
    def test_models_in_the_failed_frames_are_freed_under_the_lock(self):
        import weakref

        refs = []

        class Big:
            pass

        def deeper(model):
            raise RuntimeError("CUDA out of memory")

        @gpu.exclusive_call
        def stage():
            model = Big()
            refs.append(weakref.ref(model))
            deeper(model)

        with patch("transcribe.gpu.cuda_total_gb", return_value=3.6):
            try:
                stage()
            except RuntimeError as exc:
                held = exc  # the caller still holds the exception and its traceback
        assert held is not None
        assert refs[0]() is None
