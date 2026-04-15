# Defining a New D-GBNE Problem

This guide explains how to define a new Bayesian generalized Nash equilibrium problem in `GNE Solver`.

A D-GBNE problem in this library is built by subclassing `BayesianProblem` and implementing three methods:

- `define_players()`
- `objectives()`
- `constraints()`

These three pieces describe the full mathematical structure of the game.

## When To Use This Workflow

Use `BayesianProblem` when:

- players have type-dependent decisions
- each player may have multiple types
- you want to model type probabilities
- the problem is naturally written as a Bayesian game

If each player only has a single decision vector and there is no type structure, use `BaseProblem` instead.

## The Required Structure

A new problem class must inherit from `BayesianProblem`:

```python
from solvers.dgbne_solver import BayesianProblem

class MyBayesianProblem(BayesianProblem):
    def define_players(self):
        ...

    def objectives(self):
        ...

    def constraints(self):
        ...
```
Each method has a specific role:

- `define_players()` defines the players and their type structure
- `objectives()` returns the list of player objective functions
- `constraints()` returns the list of constraint functions

## Step 1: Define The Players
Players are represented with the `BayesianPlayer` class.

Each player typically includes:

- `name`: an optional label
- `size`: total number of decision variables controlled by the player
- `f_index`: which objective in `objectives()` belongs to that player
- `constraints`: which constraints in `constraints()` apply to that player
- `bounds`: optional variable bounds
- `type_values`: the set of possible types for that player
- `type_probs`: probabilities associated with those types
- `action_size_per_type`: number of action variables for each type
  
The total size must satisfy:
```python
size = len(type_values) * action_size_per_type
```

Example:
``` python
from solvers.dgbne_solver import BayesianPlayer
...
def define_players(self):
    P1 = BayesianPlayer(
        name="P1",
        size=2,
        f_index=0,
        constraints=[0],
        bounds=(0.0, 5.0),
        type_values=[0.0, 1.0],
        type_probs=[0.5, 0.5],
        action_size_per_type=1,
    )

    P2 = BayesianPlayer(
        name="P2",
        size=2,
        f_index=1,
        constraints=[0],
        bounds=(0.0, 5.0),
        type_values=[0.0, 1.0],
        type_probs=[0.5, 0.5],
        action_size_per_type=1,
    )

    return [P1, P2]
```
In this example:

- Player `P1` has 2 types
- Player `P2` has 2 types
- Each type controls 1 decision variable
- Each player therefore has total size `2`
- Each player has its own objective
- Both players are affected by constraint `0`

## Step 2: Understand The Player Vector Structure
In a Bayesian problem, each player's vector is still part of the full action profile:
```python
x = [x_1, x_2, ..., x_N]
```
where `x[i]` is the full flattened decision vector of player `i`.

For a Bayesian player, that vector is interpreted as stacked per-type actions. For example, if a player has:

- `type_values= = [0,1]`
- `action_size_per_type = 1`

then:
```python
x[i] = [a_i(0), a_i(1)]
```
If instead `action_size_per_type = 2`, then the vector contains both type-dependent action blocks. 

The library provides helpers such as:

- `reshape_player_actions(x_i, player_idx)`
- `expected_action(x_i, player_idx)`
- `expected_other_actions(x_structured, player_idx)`
- `type_weighted_sum(values, player_idx)`

These are useful when writing objective and constraint functions.

## Step 3: Define The Objective Functions
The `objectives()` method returns a list of functions.

Each objective function must:

- Accept the full action profile as a list of vectors
- Return a scalar

The full action profile still has the form:
```python
x = [x_1, x_2, ..., x_N]
```
where `x[i]` is the flattened Bayesian action vector of player `i`.

Example:
```python
import jax.numpy as jnp

def objectives(self):
    def f1(x):
        x1 = self.reshape_player_actions(x[0], 0)
        x2 = self.reshape_player_actions(x[1], 1)

        t1_actions = x1[:, 0]
        expected_x2 = jnp.mean(x2[:, 0])

        values = (t1_actions - 1.0) ** 2 + expected_x2
        return self.type_weighted_sum(values, 0)

    def f2(x):
        x1 = self.reshape_player_actions(x[0], 0)
        x2 = self.reshape_player_actions(x[1], 1)

        t2_actions = x2[:, 0]
        expected_x1 = jnp.mean(x1[:, 0])

        values = (t2_actions - 2.0) ** 2 + expected_x1
        return self.type_weighted_sum(values, 1)

    return [f1, f2]
```

!!! warning "Function Signature Requirement"
    Objective functions must accept a **list of player vectors** as input.

    Each entry corresponds to one player's full decision vector, so `x[i]` is the vector for player `i`.

    Every objective function must return a **scalar**.

## Step 4: Define The Constraint Functions
The `constraints()` method returns a list of constraint functions.

Each constraint function must:

- Accept the full action profile as a list of vectors
- Return a scalar

Example:
``` python
def constraints(self):
    def g1(x):
        ex1 = self.expected_action(x[0], 0)
        ex2 = self.expected_action(x[1], 1)
        return ex1[0] + ex2[0] - 3.0

    return [g1]
```
Here the shared constraint is written in terms of expected actions.

!!! warning "Constraint Signature Requirement"
    Constraint functions must accept the full action profile as a **list of player vectors**.

    Each constraint function must return a **scalar**.

## How Player Indexing Works
The player definitions must be consistent with the lists returned by `objectives()` and `constraints()`.

For example:
```python
BayesianPlayer(name="P1", size=2, f_index=0, constraints=(0,), type_values=(0, 1), type_probs=(0.5, 0.5), action_size_per_type=1)
BayesianPlayer(name="P2", size=2, f_index=1, constraints=(0,), type_values=(0, 1), type_probs=(0.5, 0.5), action_size_per_type=1)
```
means:

- Player 1 uses `objectives()[0]`
- Player 2 uses `objectives()[1]`
- Both players are affected by `constraints()[0]`

This indexing is part of the problem definition, so it is important that the returned lists and player metadata agree.

## A Complete Example
Let each player have two types, with equal probabilities. Their objective functions depend on their own type-contingent actions and on the expected action of the other player.
```python
import jax.numpy as jnp
from solvers.dgbne_solver import BayesianProblem, BayesianPlayer
from solvers.algorithms import EnergyMethod

class SimpleBayesianProblem(BayesianProblem):
    def define_players(self):
        return [
            BayesianPlayer(
                name="P1",
                size=2,
                f_index=0,
                constraints=(0,),
                bounds=(0.0, 5.0),
                type_values=[0.0, 1.0],
                type_probs=[0.5, 0.5],
                action_size_per_type=1,
            ),
            BayesianPlayer(
                name="P2",
                size=2,
                f_index=1,
                constraints=(0,),
                bounds=(0.0, 5.0),
                type_values=[0.0, 1.0],
                type_probs=[0.5, 0.5],
                action_size_per_type=1,
            ),
        ]

    def objectives(self):
        def f1(x):
            x1 = self.reshape_player_actions(x[0], 0)
            expected_x2 = self.expected_action(x[1], 1)[0]
            values = (x1[:, 0] - 1.0) ** 2 + expected_x2
            return self.type_weighted_sum(values, 0)

        def f2(x):
            x2 = self.reshape_player_actions(x[1], 1)
            expected_x1 = self.expected_action(x[0], 0)[0]
            values = (x2[:, 0] - 2.0) ** 2 + expected_x1
            return self.type_weighted_sum(values, 1)

        return [f1, f2]

    def constraints(self):
        def g1(x):
            ex1 = self.expected_action(x[0], 0)
            ex2 = self.expected_action(x[1], 1)
            return ex1[0] + ex2[0] - 3.0

        return [g1]

problem = SimpleBayesianProblem()
problem.set_initial_point(0.5, 0.1)
primal_x, dual_x = problem.solve(EnergyMethod)

print("Primal:", primal_x)
print("Dual:", dual_x)
```

## A Good Checklist
Before solving a new Bayesian problem, check that:

- `define_players()` returns a list of valid `BayesianPlayer` objects
- Each player has the correct total variable dimension
- Each player's size equals `len(type_values) * action_size_per_type`
- Each `type_probs` has the same length as `type_values`
- Each `type_probs` sums to `1.0`
- Each `f_index` matches an objective in `objectives()`
- Each constraint index matches a constraint in `constraints()`
- Each objective function accepts a `list` of player vectors
- Each constraint function accepts a `list` of player vectors
- Every objective returns a scalar
- Every constraint returns a scalar
- The initial primal and dual points have the correct sizes

## Next Step
After defining a new Bayesian problem, the next useful page is Choosing an Algorithm, where you can decide how to solve it.