# Defining a New GNE Problem

This guide explains how to define a new standard GNE problem in `GNEkit`.

A standard GNE problem in this library is built by subclassing `BaseProblem` and implementing three methods:

- `define_players()`
- `objectives()`
- `constraints()`

These three pieces describe the full mathematical structure of the game.

## When To Use This Workflow

Use `BaseProblem` when:

- each player has a single decision vector
- the game is not type-dependent
- the problem is not modeled as a Bayesian game

!!! note 
    If your problem includes types, type probabilities, or per-type actions, use the [Bayesian workflow](./dgbne_problems.md) instead.

## The Required Structure

A new problem class must inherit from `BaseProblem`:

```python
from solvers.gnep_solver import BaseProblem

class MyProblem(BaseProblem):
    def define_players(self):
        ...

    def objectives(self):
        ...

    def constraints(self):
        ...
```
Each method has a specific role:

- `define_players()` defines the players and their metadata
- `objectives()` returns the list of player objective functions
- `constraints()` returns the list of constraint functions

## Step 1: Define The Players
Players are represented with the `Player` class.

Each player typically includes:

- `name`: an optional label
- `size`: number of decision variables controlled by the player
- `f_index`: which objective in `objectives()` belongs to that player
- `constraints`: which constraints in `constraints()` apply to that player
- `bounds`: optional variable bounds

Example:
```python
from solvers.gnep_solver import Player
...
def define_players(self):
    P1 = Player(
            name="P1",
            size=1,
            f_index=0,
            constraints=[0],
            bounds=(0.0, 5.0),
        )
    P2 = Player(
            name="P2",
            size=1,
            f_index=1,
            constraints=[0],
            bounds=(0.0, 5.0),
        )
    return [P1, P2]

```
In this example:

- player `P1` controls one variable
- player `P2` controls one variable
- each player has its own objective
- both players are affected by constraint `0`

!!! note
    Players can also be defined in batch. See the [API Docs](../reference/players/base_player.md).

## Step 2: Define The Objective Functions
The `objectives()` method returns a list of functions.

Each objective function must:

- Accept the full action profile as a list of vectors
- Return a scalar
- The full action profile has the form:
```python
x = [x_1, x_2, ..., x_N]
```
where `x[i]` is the decision vector of player `i`.

Example:
```python
def objectives(self):
    def f1(x):
        x1, x2 = x
        return (x1[0, 0] - 1.0) ** 2 + x2[0, 0]

    def f2(x):
        x1, x2 = x
        return (x2[0, 0] - 2.0) ** 2 + x1[0, 0]

    return [f1, f2]
```

!!! warning "Function Signature Requirement"
    Objective functions must accept a **list of player vectors** as input.

    Each entry corresponds to one player's variables, so `x[i]` is the vector for player `i`.

    Every objective function must return a **scalar**.

## Step 3: Define The Constraint Functions
The `constraints()` method returns a list of constraint functions.

Each constraint function must:

- Accept the full action profile as a list of vectors
- Return a scalar

Example:

For the inequality $x_1 + x_2 \leq 3$, we code it as:
```python
def constraints(self):
    def g1(x):
        x1, x2 = x
        return x1[0, 0] + x2[0, 0] - 3.0

    return [g1]
```

!!! warning "Constraint Signature Requirement"
    Constraint functions must accept the full action profile as a **list of player vectors**.

    Each constraint function must return a **scalar**.


## How Player Indexing Works
The Player definitions must be consistent with the lists returned by `objectives()` and `constraints()`.

For example:
```python
Player(name="P1", size=1, f_index=0, constraints=(0,))
Player(name="P2", size=1, f_index=1, constraints=(0,))
```
means:

- Player 1 uses `objectives()[0]`
- Player 2 uses `objectives()[1]`
- Both players are affected by `constraints()[0]`
  
This indexing is part of the problem definition, so it is important that the returned **lists and player metadata agree**.


## A Complete Example
Let $f_1, f_2$ denote the objective functions for player 1 and 2, respectively.
$$
f_1(x) = (x_1 - 1)^2 + x_2 \qquad f_2(x) = (x_2 - 2)^2 + x_1
$$
They each control one variable and share the following constraint:
$$
x_1 + x_2 \leq 3, \qquad x_1, x_2 \geq 0
$$

```python
from solvers.gnep_solver import BaseProblem, Player
from solvers.algorithms import EnergyMethod

class SimpleProblem(BaseProblem):
    def define_players(self):
        return [
            Player(name="P1", size=1, f_index=0, constraints=(0,), bounds=(0.0, 5.0)),
            Player(name="P2", size=1, f_index=1, constraints=(0,), bounds=(0.0, 5.0)),
        ]

    def objectives(self):
        def f1(x):
            x1, x2 = x
            return (x1[0, 0] - 1.0) ** 2 + x2[0, 0]

        def f2(x):
            x1, x2 = x
            return (x2[0, 0] - 2.0) ** 2 + x1[0, 0]

        return [f1, f2]

    def constraints(self):
        def g1(x):
            x1, x2 = x
            return x1[0, 0] + x2[0, 0] - 3.0

        return [g1]

problem = SimpleProblem()
problem.set_initial_point(1.0, 0.1)
primal_x, dual_x = problem.solve(EnergyMethod)

print("Primal:", primal_x)
print("Dual:", dual_x)
```

## A Good Checklist
Before solving a new problem, check that:

- `define_players()` returns a `list` of valid `Player` objects
- Each player has the correct variable dimension
- Each f_index matches an objective in `objectives()`
- Each constraint index matches a constraint in `constraints()`
- Each objective function accepts a `list` of player vectors
- Each constraint function accepts a `list` of player vectors
- Every objective returns a scalar
- Every constraint returns a scalar
- The initial primal and dual points have the correct sizes

## Next Step
After defining a new problem, the next useful page is `Choosing an Algorithm`, where you can decide how to solve it.