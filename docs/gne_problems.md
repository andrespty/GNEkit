# Generalized Nash Equilibrium Problems

We provide an introduction to the mathematical formulation of Generalized Nash Equilibrium problems and connect it to the abstractions used in `GNEkit`.

## Overview

A Generalized Nash Equilibrium Problem (GNEP) is an extension of the classical Nash Equilibrium Problem (NEP) where each player's feasible strategy set depends on the strategies chosen by the other players — not just their own. This dependence is what makes the problem *generalized*.

To understand why this matters, consider first the classical NEP: each player chooses a strategy from a fixed set, taking the other players' decisions as given, and tries to minimize their own objective. The sets do not interact — player $i$ is always allowed to play any strategy in $X_i$, regardless of what others do.

In a GNEP, this is no longer the case. What player $i$ is *allowed* to do may depend on what the other players are doing. This coupling between players' feasible sets is the defining feature of the GNEP, and it is what makes these problems significantly harder to analyze and solve.

## Basic Setting

Consider a game with $N$ players. For each player $i \in \{1, \dots, N\}$:

- $x_i \in \mathbb{R}^{n_i}$ denotes the decision vector of player $i$
- $x_{-i}$ denotes the collection of decisions of all players except player $i$
- $x = (x_1, \dots, x_N)$ denotes the full strategy profile of all players
- $f_i(x_i, x_{-i})$ denotes the objective function of player $i$
- $X_i(x_{-i})$ denotes the feasible set of player $i$, which depends on what the other players are doing

The goal of player $i$ is to solve:

$$
\min_{x_i} \; f_i(x_i, x_{-i})
\qquad \text{subject to} \qquad x_i \in X_i(x_{-i})
$$

This is just a standard optimization problem — but it is parameterized by $x_{-i}$. As the other players change their decisions, both the objective and the feasible set of player $i$ may change.

## Generalized Nash Equilibrium

A strategy profile $x^* = (x_1^*, \dots, x_N^*)$ is called a **Generalized Nash Equilibrium (GNE)** if no player can unilaterally improve their objective while remaining feasible, given the strategies of the other players.

More precisely, $x^*$ is a GNE if for every player $i$:

$$
x_i^* \in \arg\min_{x_i \in X_i(x_{-i}^*)} f_i(x_i, x_{-i}^*)
$$

This is equivalent to requiring that $x_i^*$ is feasible and optimal for player $i$:

$$
x_i^* \in X_i(x_{-i}^*)
\qquad \text{and} \qquad
f_i(x_i^*, x_{-i}^*) \leq f_i(x_i, x_{-i}^*)
\quad
\text{for all } x_i \in X_i(x_{-i}^*)
$$

Note that the GNE is generally not unique. In fact, GNEPs with shared constraints (introduced below) often admit a continuum of equilibria. This multiplicity is one of the key challenges the GNEP literature addresses, and it motivates refined solution concepts such as the *variational equilibrium*, discussed later.

## Shared Constraints

A very common and important special case is when the coupling between players arises from a **shared constraint** — a constraint that involves all players' decisions simultaneously and must be satisfied by the group as a whole.

In this setting, each player solves:

$$
\min_{x_i} \; f_i(x_i, x_{-i})
\qquad \text{subject to} \qquad g(x_i, x_{-i}) \leq 0
$$

possibly alongside player-specific constraints $x_i \in X_i^0$. The feasible set of player $i$ is then:

$$
X_i(x_{-i}) = \{x_i \in X_i^0 : g(x_i, x_{-i}) \leq 0\}
$$

Notice how $X_i(x_{-i})$ depends on $x_{-i}$ through the shared constraint — this is the source of the coupling.

A particularly well-studied subclass is the **jointly convex GNEP**, where the shared constraint can be written as $g(x) \leq 0$ with $g$ convex in the full vector $x = (x_1, \dots, x_N)$. In this case there is a single joint feasible set $\mathcal{X} = \{x : g(x) \leq 0\}$, and the feasible region for each player is just a slice of $\mathcal{X}$ at fixed $x_{-i}$. This structure enables more tractable analysis and is the setting most commonly targeted by solvers in the literature. However, `GNEkit` is not limited to this case — it handles the general GNEP, where $g$ may be non-convex or depend on $x_{-i}$ in more complex ways.

Shared constraints arise naturally in many applications:

- **Network resource allocation**: users share bandwidth or capacity
- **Congestion games**: players share infrastructure (roads, servers)
- **Power systems**: generators share transmission limits
- **Economic equilibrium models**: agents share a market-clearing constraint