# Generalized Nash Equilibrium Problems

We provide an introduction to the mathematical formulation of Generalized Nash Equilibrium problems and connect it to the abstractions used in `GNEkit`.

## Overview

In the classical Nash Equilibrium Problem (NEP), each player has a fixed set of strategies they are allowed to play, and they choose from that set to minimize their own objective. The key word is *fixed*: what player $i$ is allowed to do does not depend on what others do.

A **Generalized Nash Equilibrium Problem (GNEP)** relaxes this. Each player still minimizes their own objective, but their **feasible set can depend on the other players' decisions**. This single change — coupling the players' feasible sets — is what defines the GNEP and makes it substantially harder to analyze and solve.

## Formal Setup

Consider a game with $N$ players. Player $i$ has a decision vector $x_i\in \mathbb{R}^{n_i}$. We write $x_{-i}$ for the decisions of all players except $i$, and $x=(x_1,...,x_N)$ for the full strategy profile. 

Each player $i$ solves:
$$
\min_{x_i} f_i(x_i,x_{-i}) \quad \text{subject to }\quad x_i \in X_i(x_{-i})
$$
where $f_i$ is player $i$'s objective and $X_i(x_{-i})$ is their feasible set, which is allowed to depend on the other players' decisions. This is what separates a GNEP from a NEP: in a NEP, $X_i$ is a constant set, here it is a function of $x_{-i}$$^{2}$.

## Generalized Nash Equilibrium

A strategy profile $x^* = (x_1^*, \dots, x_N^*)$ is called a **Generalized Nash Equilibrium (GNE)** if no player can unilaterally reduce their objective while staying feasible, given what everyone else is doing. 

More precisely, for every player $i$:

$$
x_i^* \in \arg\min_{x_i \in X_i(x_{-i}^*)} f_i(x_i, x_{-i}^*)
$$

This is a fixed-point condition: each player is best-responding to the others, and the others' strategies are consistent with their own best responses.

Unlike classical Nash equilibria, GNEs are generally not unique. GNEPs often admit a continuum of equilibria, which creates both theoretical and computational challenges. Selecting among them requires additional criteria, discussed in [Theory]()


## Constraints
In practice, the dependence of $X_i(x_{-i})$ on other players almost always arises from **constraints that couple the players' decisions**. The most common structure is:

$$
X_i(x_{-i}) = \{x_i \in X_i^0: g_i(x_i, x_{-i}) \leq 0\}
$$
where $X_i^0$ is a player-specific private constraint set and $g_i$ is a constraint function that involves other players' decisions. The constraint $g_i$ may involve all other players, a subset of them, or encode asymmetric relationships. There is no requirement that every player faces the same constraints.

A widely studied special case is the **jointly convex GNEP**, where all players share the *same* constraint: $g_1=...=g_n=g$, with $g$ convex in the full vector $x$. This defines a single joint feasible region $X=\{x: g(x) \leq 0\}$, and each player's feasible set is a slice of it at fixed $x_{-i}$. The symmetry of this structure (everyone faces the same rule) makes the problem more tractable and enables refined solution concepts such as the *variational equilibrium*. Most algorithms in the literature are designed for this case.

`GNEkit` handles the general GNEP, including asymmetric and non-convex coupling, not just the jointly convex case.

## Examples

Shared constraints arise naturally in many applications:

- **Network resource allocation**: users share bandwidth or capacity
- **Congestion games**: players share infrastructure (roads, servers)
- **Power systems**: generators share transmission limits
- **Economic equilibrium models**: agents share a market-clearing constraint 

## References
1. Arrow, K. J., and Debreu, G. *Existence of an equilibrium for a competitive economy*.  
   [https://doi.org/10.2307/1907353](https://doi.org/10.2307/1907353)

2. Rosen, J. B. *Existence and uniqueness of equilibrium points for concave N-person games*.  
   [https://doi.org/10.2307/1911749](https://doi.org/10.2307/1911749)

3. Facchinei, F., and Kanzow, C. *Generalized Nash equilibrium problems*.  
   [https://doi.org/10.1007/s10479-009-0653-x](https://doi.org/10.1007/s10479-009-0653-x)

4. Facchinei, F., Fischer, A., and Piccialli, V. *On generalized Nash games and variational inequalities*.  
   [https://doi.org/10.1016/j.orl.2006.03.004](https://doi.org/10.1016/j.orl.2006.03.004)

5. Kulkarni, A. A., and Shanbhag, U. V. *On the variational equilibrium as a refinement of the generalized Nash equilibrium*.  
   [https://doi.org/10.1016/j.automatica.2011.09.042](https://doi.org/10.1016/j.automatica.2011.09.042)

6. Hobbs, B. F., and Pang, J. S. *Nash-Cournot equilibria in electric power markets with piecewise linear demand functions and joint constraints*.  
   [https://doi.org/10.1287/opre.1060.0342](https://doi.org/10.1287/opre.1060.0342)