"""Characterization tests: the restructure must not change any of these numbers.

The expected values are in tests/baseline/baseline.json, produced by
tests/baseline/generate_baseline.py from the code before the restructure.
"""
import json
import math
import pathlib

import pytest

from _compat import PROBLEMS, ALGORITHMS
from snapshot import problem_snapshot, algorithm_snapshot

BASELINE = json.loads((pathlib.Path(__file__).parent / "baseline" / "baseline.json").read_text())
RTOL, ATOL = 1e-9, 1e-12


def assert_close(actual, expected, where):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), where
        for k in expected:
            assert_close(actual[k], expected[k], f"{where}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), where
        for k, (a, e) in enumerate(zip(actual, expected)):
            assert_close(a, e, f"{where}[{k}]")
    elif isinstance(expected, float):
        if math.isnan(expected):
            assert math.isnan(actual), where
        else:
            assert actual == pytest.approx(expected, rel=RTOL, abs=ATOL), where
    else:
        assert actual == expected, where


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_problem_snapshot(name):
    snap = json.loads(json.dumps(problem_snapshot(PROBLEMS[name]())))
    assert_close(snap, BASELINE[name]["problem"], name)


@pytest.mark.parametrize("alg_name", sorted(ALGORITHMS))
@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_algorithm_snapshot(name, alg_name):
    expected = BASELINE[name]["algorithms"][alg_name]
    if "error" in expected:
        with pytest.raises(Exception) as info:
            algorithm_snapshot(PROBLEMS[name](), ALGORITHMS[alg_name])
        assert type(info.value).__name__ == expected["error"]
    else:
        actual = algorithm_snapshot(PROBLEMS[name](), ALGORITHMS[alg_name])
        assert_close(actual, expected, f"{name}.{alg_name}")
