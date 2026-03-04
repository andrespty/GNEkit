from typing import List, Optional

class Player:
    def __init__(self,
                 name: Optional[str] = None,
                 action_size: Optional[int] = None,
                 obj_func: int = None,
                 constraints:List[int] = None,
                 bounds = None
        ):
        self.name = name
        self.size = action_size             # Which part of the global vector is theirs
        self.f_index = obj_func             # The function to minimize
        self.constraints = constraints      # Local g_i(x_i, x_{-i}) <= 0
        self.bounds = bounds