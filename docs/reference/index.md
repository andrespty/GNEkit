# API Reference

### Problems

- [`BaseProblem`](./problems/base_problem.md) is the base abstraction for standard GNE problems.
- [`BayesianProblem`](./problems/bayesian_problem.md) extends the same workflow to discrete Bayesian settings with type-dependent actions.

### Players

- [`Player`](./players/base_player.md) describes a standard player: variable size, objective index, shared constraints, and optional bounds.
- [`BayesianPlayer`](./players/bayesian_player.md) adds type values, type probabilities, and per-type action structure.

### Algorithms

- [`BaseAlgorithm`](./algorithms/base_algorithm.md) defines the common solver interface and shared setup.
- [`EnergyMethod`](./algorithms/energy_method.md) is the primary documented solver implementation and the default choice in most examples.

## Utility Functions

These helpers support problem setup and solver internals:

- [`construct_vectors`](./functions/construct_vectors.md) splits a stacked action vector into per-player blocks.
- [`flatten_variables`](./functions/flatten_variables.md) flattens structured variables into a single solver vector.
- [`create_wrapped_function`](./functions/create_wrapped_function.md) freezes opponents' actions to build single-player objective views.
- [`one_hot_encoding`](./functions/one_hot_encoding.md) builds the player-to-function incidence matrix used by the solver.
- [`players_to_lists`](./functions/players_to_lists.md) converts standard player objects into parallel attribute lists.
- [`bayesian_players_to_lists`](./functions/bayesian_players_to_lists.md) does the same for Bayesian players.

## Types

- [`Object Types`](./types/object_types.md) summarizes the main array, function, and structural type aliases used across the package.

## Related Guides

If you want implementation walkthroughs rather than API details, start here:

- [`Core Concepts`](../guides/core_concepts.md)
- [`Create New GNE Problems`](../guides/gne_problems.md)
- [`Create New D-GBNE Problems`](../guides/dgbne_problems.md)
- [`Creating an Algorithm`](../guides/creating_algorithm.md)
