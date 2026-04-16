# Core Concepts

`GNEkit` is built around a small set of reusable abstractions for modeling and solving generalized Nash equilibrium problems.

Here, we explain the main ideas behind the library so the rest of the documentation feels more natural.

## The Main Workflow

Most usage follows the same pattern:

1. Define or import a problem
2. Define the players in that problem
3. Define the objective and constraint functions
4. Choose an algorithm
5. Set an initial point
6. Solve for the equilibrium variables

In practice, the library is designed so that a problem class holds the mathematical structure, while an algorithm class handles the numerical solution.

## Players

A `Player` represents one decision-maker in the game.

Each player stores the information needed to describe that player’s role in the equilibrium problem, such as:

- Player name
- Number of decision variables they control
- Which objective function belongs to them
- Which constraints apply to them
- Optional variable bounds

This makes the problem definition explicit and reusable.

For D-GBNE problems, the library uses `BayesianPlayer`, which extends the same idea with type-dependent information such as:

- type values
- type probabilities
- action size per type

## Problems

A problem class defines the equilibrium model itself.

The main standard abstraction is `BaseProblem`. A subclass of `BaseProblem` is responsible for describing:

- Players in the game
- Objective functions
- Constraint functions

For Bayesian settings, the library provides `BayesianProblem`, which extends the standard problem interface with helpers for type-structured actions and expected values.

In other words:

- `Player` describes who is acting
- `BaseProblem` describes the game they are playing

## Objective and Constraint Functions

A Generalized Nash Equilibrium problem is built from:

- Objective functions for the players
- Shared or player-specific constraints

In this library, objectives and constraints are defined as callables over the full action profile.

That means each function receives the current actions of all players and returns a scalar value.

!!! warning "Function Signature Requirement"

    Objective and constraint functions must accept a **list of vectors** representing the full action profile.

    Each entry in the list corresponds to one player’s decision vector, so `x[i]` contains the variables of player `i`.

    Every objective function and every constraint function must return a **scalar**.



## Primal And Dual Variables

The solver works with two kinds of variables:

- primal variables
- dual variables

The primal variables are the players’ actual decision variables.

The dual variables correspond to the constraints and act like multipliers in the constrained optimization formulation.

Before solving a problem, you provide both through `set_initial_point(primal_x, dual_x)`.

## Algorithms

Algorithms are responsible for computing a solution once the problem has been defined.

The current architecture separates the problem definition from the numerical method. This makes it possible to reuse the same problem with different algorithms.

The main algorithms currently exposed are:

- `EnergyMethod`
- `FBEnergyMethod`

These are passed into the problem’s `solve(...)` method as algorithm classes.

## GNE and D-GBNE Problems

The library supports two related workflows.

Standard GNE problems use:

- `Player`
- `BaseProblem`

D-GBNE problems use:

- `BayesianPlayer`
- `BayesianProblem`

Both share the same general solving pattern, which keeps the library consistent across problem families.

## Reusable Problem Classes

The `problems/` package contains predefined problem classes that follow these abstractions.

These are useful for:

- Learning the library
- Running benchmark-style examples
- Testing algorithms
- Building new problem classes by analogy

The `solvers/` package contains the reusable infrastructure that these problem classes rely on.

## Mental Model

A helpful way to think about the package is:

- Players define the variable structure
- Problems define the mathematical model
- Algorithms define the numerical solution method
- Initial points provide the starting state
- Solving returns primal and dual equilibrium variables

## Where To Go Next

After this page, a good next step is to read the [User Guide](../guides/user_guide.md) for the practical workflow, or open the [API Reference](../reference/index.md) if you want class-level details.
