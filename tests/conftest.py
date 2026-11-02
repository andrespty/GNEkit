import pathlib
import sys

TESTS = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(TESTS), str(TESTS.parent / "src")]
