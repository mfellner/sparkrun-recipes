"""Task 3 CPU shape contracts; no CUDA graph correctness claim."""
from test_drama3_compact_head import runtime
import pytest


@pytest.mark.parametrize("capacity", [128, 256, 513, 10240])
def test_smaller_buckets_gated_and_clamped(capacity, monkeypatch):
    perf = runtime()
    assert perf.graph_buckets(capacity) == tuple(sorted({min(b, capacity) for b in (1024, 2048, 4096, 10240)}))
    monkeypatch.setenv("FISH_DRAMA3_SMALL_GRAPH_BUCKETS", "1")
    buckets = perf.graph_buckets(capacity)
    assert buckets == tuple(sorted({min(b, capacity) for b in (256, 512, 1024, 2048, 4096, 10240)}))
    assert buckets[-1] == capacity
    for position in range(capacity - 1):
        selected = perf.select_bucket(buckets, position)
        assert selected >= position + 2
        assert selected == min(b for b in buckets if b >= position + 2)
    with pytest.raises(ValueError):
        perf.select_bucket(buckets, capacity - 1)
    with pytest.raises(ValueError):
        perf.select_bucket(buckets, -1)


def test_small_bucket_boundary_position_plus_two(monkeypatch):
    perf = runtime()
    monkeypatch.setenv("FISH_DRAMA3_SMALL_GRAPH_BUCKETS", "1")
    buckets = perf.graph_buckets(10240)
    assert [perf.select_bucket(buckets, p) for p in (254, 255, 510, 511, 10238)] == [256, 512, 512, 1024, 10240]
