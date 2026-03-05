import inspect
from functools import wraps
from typing import List, Callable


def validate_math_functions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        items = func(*args, **kwargs)

        # 1. Ensure it's a list
        if not isinstance(items, list) or len(items) == 0:
            raise TypeError(f"{func.__name__} must return a non-empty list of callables.")

        for i, item in enumerate(items):
            # 2. Ensure it's actually callable
            if not callable(item):
                raise TypeError(f"Item {i} in {func.__name__} is not a function/callable.")

            # 3. Check signature (Must have exactly 1 parameter)
            sig = inspect.signature(item)
            params = sig.parameters
            if len(params) != 1:
                raise ValueError(
                    f"Function {item.__name__ if hasattr(item, '__name__') else i} "
                    f"in {func.__name__} must have exactly 1 parameter. Found: {len(params)}"
                )
        return items

    return wrapper