class PlayerValidator:
    def validate(self, name,size,f_index,constraints,bounds):
        if name is not None and not isinstance(name, str):
            raise TypeError("Player name must be a string or None.")

        if not isinstance(size, int) or size <= 0:
            raise ValueError("size must be a positive integer.")

        if not isinstance(f_index, int) or f_index < 0:
            raise ValueError("f_index must be >= 0.")

        for c in constraints:
            if c is not None and (not isinstance(c, int) or c < 0):
                raise ValueError("constraints must be integers >= 0 or None.")

        if bounds is not None:
            if isinstance(bounds, tuple) and len(bounds) == 2:
                self._validate_bound_pair(bounds)

            elif isinstance(bounds, (tuple, list)):
                if len(bounds) != size:
                    raise ValueError(f"Player {name} has {size} variables but {len(bounds)} bounds.")
                for b in bounds:
                    self._validate_bound_pair(b)
            else:
                raise TypeError("bounds must be a tuple (lb, ub) or a list of tuples.")


    @staticmethod
    def _validate_bound_pair(b):
        if not isinstance(b, tuple) or len(b) != 2:
            raise TypeError("Each bound must be a tuple (lower, upper).")
        lower, upper = b
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise TypeError("Bounds must be numeric.")
        if lower >= upper:
            raise ValueError(f"Lower bound {lower} must be less than upper bound {upper}.")
