<div class="hero">
  <h1>Generalized Nash Kit</h1>
  <p>
    A Python library for modeling and solving Generalized Nash Equilibrium problems,
    including Bayesian variants, in a reusable and research-friendly way.
  </p>
</div>

<div class="grid cards">

<a href="./getting_started/installation" class="card">
  <h3>Getting Started</h3>
  <p>Install the package and run your first example.</p>
</a>

<a href="./guides/core_concepts" class="card">
  <h3>Guides</h3>
  <p>Learn how to define standard and Bayesian problems.</p>
</a>

<a href="./reference/" class="card">
  <h3>API Reference</h3>
  <p>Explore the core classes, algorithms, and utilities.</p>
</a>

<a href="./examples/examples" class="card">
  <h3>Examples</h3>
  <p>See concrete problem setups and workflows.</p>
</a>
</div>

## Key Features

- **Benchmark problem collection**  
  Includes a broad collection of Generalized Nash equilibrium problems drawn from the literature, making it easier to reproduce and compare algorithmic results.

- **Simple problem-definition workflow**  
  Provides a clear and reusable class structure for implementing new problems and solving them with the available algorithms.

- **Algorithm experimentation**  
  Makes it straightforward to plug in, test, and evaluate new solution algorithms against a shared problem interface.

- **Support for D-GBNE models**  
  Supports Discrete Generalized Bayesian Nash Equilibrium (D-GBNE) problems through dedicated Bayesian player and problem abstractions.

- **Utility functions for development**  
  Includes helper functions for vector construction, indexing, flattening, and other common tasks that arise when defining problems and solver workflows.


## Who This is For
This project is especially useful for:

- Researchers working on Generalized Nash Equilibrium problems
- Students learning equilibrium problem formulations
- Developers building reusable game-theoretic problem classes
- Anyone experimenting with constrained multi-player optimization