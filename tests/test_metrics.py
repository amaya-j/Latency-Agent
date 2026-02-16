from latency_agent.metrics import percentile


def test_percentile_basic():
    xs = [1, 2, 3, 4, 5]
    assert percentile(xs, 0) == 1
    assert percentile(xs, 50) == 3
    assert percentile(xs, 100) == 5


def test_percentile_singleton():
    assert percentile([42], 99.9) == 42
