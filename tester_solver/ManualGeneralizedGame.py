from gnep_solver import *
from gnep_solver.utils import one_hot_encoding
from typing import List, Callable
from scipy.optimize import basinhopping
import timeit
from gnep_solver.Player import Player, players_to_lists
import jax
import jax.numpy as jnp
from jax import eval_shape
from gnep_solver.EnergyMethod import EnergyMethod
from gnep_solver.GeneralizedGame import GeneralizedGame
jax.config.update("jax_enable_x64", True)

class ManualGeneralizedGame(GeneralizedGame):
    def __init__(
            self,
            obj_funcs: List[ObjFunction],
            obj_funcs_der: List[ObjFunction],
            constraints: List[ConsFunction],
            constraints_der: List[ConsFunction],
            player_list: List[Player]
        ):
        super().__init__(obj_funcs, constraints, player_list)

        # 1. Validate Objective Derivatives
        self._validate_manual_derivatives(obj_funcs_der, "Objective")
        self.obj_derivatives = self._wrap_manual_functions(obj_funcs_der, is_obj=True)

        # 2. Validate Constraint Derivatives
        self._validate_manual_derivatives(constraints_der, "Constraint")
        self.const_derivatives = self._wrap_manual_functions(constraints_der, is_obj=False)

        # 3. Re-instantiate the solver with the NEW manual derivatives
        self.solver = EnergyMethod(
            self.action_sizes,
            self.obj_derivatives,
            self.const,
            self.const_derivatives,
            self.player_obj_idx,
            self.player_const_idx,
            self.bounds
        )

    def _wrap_manual_functions(self, manual_funcs, is_obj=True):
        wrapped_funcs = []

        for func_idx, user_func in enumerate(manual_funcs):
            if is_obj:
                # Identify which players actually "own" or use this function
                # Find all players whose f_index matches this function index
                active_players = [i for i, p in enumerate(self.players) if p.f_index == func_idx]
                def padded_grad(x_structured, u_func=user_func):
                    # 1. Call the user's "easy" manual derivative
                    # The user returns a dict or list: {player_idx: grad_vector}
                    user_dict_grads = u_func(x_structured)

                    # 2. Build the full list that EnergyMethod expects
                    return [
                        user_dict_grads[i] if i in user_dict_grads else jnp.zeros((size,))
                        for i, size in enumerate(self.action_sizes)
                    ]
                # wrapped_funcs.append(jax.jit(padded_grad))
                wrapped_funcs.append(padded_grad)
            else:
                # wrapped_funcs.append(jax.jit(user_func))
                wrapped_funcs.append(user_func)

        return wrapped_funcs

    def _validate_manual_derivatives(self, der_list: List[Callable], label: str):
        """Ensures manual derivatives return the correct structure and shapes."""
        if len(der_list) != (len(self.obj_functions) if label == "Objective" else len(self.const)):
            raise ValueError(f"Number of {label} derivatives must match number of functions.")

        # Create a dummy structured input
        dummy_x = construct_vectors(jnp.zeros((sum(self.action_sizes),)), self.action_sizes)

        for i, der_func in enumerate(der_list):
            try:
                out = der_func(dummy_x)

                if label == "Objective":
                    # Expecting Sparse Dictionary: {player_idx: grad_array}
                    if not isinstance(out, dict):
                        raise TypeError(f"Objective derivative {i} must return a dict of index: grad.")
                    # Check only the provided indices
                    for p_idx, grad_comp in out.items():
                        if jnp.shape(grad_comp) != (self.action_sizes[p_idx],):
                            raise ValueError(f"Obj {i}: Player {p_idx} grad shape mismatch.")

                else:
                    if not isinstance(out, (list, tuple)) or len(out) != len(self.action_sizes):
                        raise ValueError(f"Constraint {i} must return a list of length {len(self.action_sizes)}.")

                    for p_idx, grad_comp in enumerate(out):
                        if jnp.shape(grad_comp) != (self.action_sizes[p_idx],):
                            raise ValueError(f"Cons {i}: Player {p_idx} grad shape mismatch.")
            except Exception as e:
                raise ValueError(f"Validation failed for {label} derivative {i}: {e}")