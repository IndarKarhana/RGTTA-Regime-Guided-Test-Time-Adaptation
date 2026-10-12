"""Device routing must not silently turn paid GPU experiments into CPU runs."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "src"))
from run_unified_benchmark import UnifiedBenchmark


def test_cuda_fails_without_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="refusing CPU fallback"):
        UnifiedBenchmark(results_dir=str(tmp_path), device="cuda")


def test_single_gpu_rejects_multiple_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="workers 1"):
        UnifiedBenchmark(results_dir=str(tmp_path), device="cuda", n_workers=2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_reaches_forecasters(tmp_path, monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    def require_device_index(selected):
        assert selected.index is not None
    monkeypatch.setattr(torch.cuda, "set_device", require_device_index)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "test GPU")
    bench = UnifiedBenchmark(results_dir=str(tmp_path), device=device,
                             policies=["retrain", "tta", "ewc", "dynatta", "rgtta", "rgtta_ewc", "rgtta_dynatta"])
    models = bench._build_forecasters("itransformer", "synth_recurring", 96, 0)
    assert all(fc.device.type == device for fc in models.values())
    assert bench.results["metadata"]["device"] == ("cuda:0" if device == "cuda" else "cpu")


def test_cuda_timing_synchronizes(tmp_path, monkeypatch):
    bench = UnifiedBenchmark(results_dir=str(tmp_path), device="cpu")
    bench.device = torch.device("cuda")
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", calls.append)
    assert bench._clock() > 0
    assert calls == [torch.device("cuda")]
