"""Regenerate tests/baseline/baseline.json from the current code.

Run only when a numerical change is intended:
    python tests/baseline/generate_baseline.py
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]

from _compat import PROBLEMS, ALGORITHMS  # noqa: E402
from snapshot import problem_snapshot, algorithm_snapshot  # noqa: E402


def main():
    data = {}
    for name, cls in PROBLEMS.items():
        problem = cls()
        entry = {"problem": problem_snapshot(problem), "algorithms": {}}
        for alg_name, Alg in ALGORITHMS.items():
            try:
                entry["algorithms"][alg_name] = algorithm_snapshot(problem, Alg)
            except Exception as exc:  # recorded so the test can check it still fails the same way
                entry["algorithms"][alg_name] = {"error": type(exc).__name__}
        data[name] = entry
        print(name, "ok")
    (HERE / "baseline.json").write_text(json.dumps(data, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
