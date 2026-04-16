# Assumptions and Theoretical Results

This page summarizes the existence and uniqueness results for the D-GBNE.
Proofs are omitted; references are provided for each result.

---

## Assumptions

The theoretical results rely on the following assumptions, grouped by what
they are needed for.

**Assumption 1 (Local action sets).** For each player $i \in \mathcal{N}$ and
type $\theta_i \in \Theta_i$, the action set $A_i(\theta_i)$ is nonempty, convex,
and compact.

**Assumption 2 (Continuity and convexity).** For each player $i$ and type
$\theta_i$, the utility $u_i$ and constraint $g_i$ are continuous in the full
action profile $a$. Furthermore, $u_i$ is concave in $a_i$ and $g_i$ is convex
in $a_i$.

**Assumption 3 (Slater condition).** For every player $i$, type $\theta_i$, and
rival strategy $a_{-i}$, there exists a strictly feasible action
$\bar{a}_i \in A_i(\theta_i)$ such that $h_i(\bar{a}_i, a_{-i} \mid \theta_i) < 0$.

**Assumption 4 (Smoothness).** For each player $i$ and type profile $\theta \in \Theta$,
the utility $u_i(a, \theta)$ is twice continuously differentiable in $a$.

**Assumption 5 (Joint convexity of constraints).** For each player $i$ and type
profile $\theta \in \Theta$, the constraint function $g_i(a, \theta)$ is convex in
the **full** action profile $a = (a_i, a_{-i})$, not just in $a_i$.

> Assumption 5 strengthens the convexity requirement in Assumption 2, which
> only requires convexity of $g_i$ in player $i$'s own action. This stronger
> condition is what makes the lifted GNEP jointly convex, enabling the
> variational inequality characterization.

**Assumption 6 (Negative definiteness of payoff Jacobian).** The Jacobian of
the payoff gradient,

$$
J_\Upsilon(a) = \begin{pmatrix}
\nabla^2_{a_1 a_1} \Upsilon_1 & \cdots & \nabla^2_{a_1 a_N} \Upsilon_1 \\
\vdots & \ddots & \vdots \\
\nabla^2_{a_N a_1} \Upsilon_N & \cdots & \nabla^2_{a_N a_N} \Upsilon_N
\end{pmatrix}
$$

is negative definite for all $a \in \mathcal{K}$.

---

## Existence

**Theorem (Existence of D-GBNE).** *Under Assumptions 1–3, the D-GBNE
admits at least one solution $s^*$.*

The proof uses the GNEP reformulation. By lifting each player's strategy into
an extended action vector $a_i \in \mathcal{A}_i$, the D-GBNE becomes a
deterministic GNEP in a finite-dimensional space. The lifted action space
$\mathcal{A} = \prod_{i} \mathcal{A}_i$ is nonempty, compact, and convex.
The best-response correspondence of each player inherits upper hemicontinuity
with nonempty, compact, and convex values from the continuity and concavity
in Assumption 2 and the Slater condition in Assumption 3. Existence then
follows from Kakutani's fixed-point theorem applied to the joint best-response
map [1, 2].

This mirrors the classical existence argument for GNEPs, with the key
observation that the type structure introduces no additional difficulty: the
ex-ante objectives $\Upsilon_i$ and interim feasible sets $\mathcal{K}_i(a_{-i})$
inherit all the required regularity from the underlying $u_i$ and $g_i$.

---

## Uniqueness

As in the GNEP setting, GNEs are generically non-isolated when shared
constraints are active. The uniqueness result therefore targets the
**variational equilibrium (VE)** — the GNE whose Lagrange multipliers for
the shared constraints are identical across all players.

**Theorem (Uniqueness of variational D-GBNE).** *Under Assumptions 1–6,
the D-GBNE admits a unique variational equilibrium.*

Under Assumption 5, the lifted GNEP is jointly convex and the joint feasible
set

$$
\mathcal{K} = \bigl\{ a \in \mathcal{A} : h_i(a_i^{\theta_i}, a_{-i} \mid \theta_i) \leq 0,\;
\forall i \in \mathcal{N},\; \forall \theta_i \in \Theta_i \bigr\}
$$

is convex. The VE is then the solution to $\text{VI}(\mathcal{K}, -\nabla_a \Upsilon)$.
Assumption 6 — negative definiteness of $J_\Upsilon$ — implies that the map
$a \mapsto -\nabla_a \Upsilon(a)$ is strictly monotone over $\mathcal{K}$, which
rules out two distinct solutions to the VI [3, 4].

---

## Relation to Classical Results

The D-GBNE results generalize two classical lines of work:

- **Rosen (1965)**: in the complete information case ($m_i = 1$ for all $i$),
  the payoff Jacobian $J_\Upsilon$ reduces to the standard pseudo-gradient
  Jacobian of a GNEP, and Assumption 6 recovers Rosen's diagonal strict
  convexity condition. Uniqueness of the VE then reduces to Rosen's original
  result.

- **Ui (2016)**: in the absence of coupled constraints ($\mathcal{K} = \mathcal{A}$),
  the VI reduces to the discrete-type Bayesian Nash equilibrium
  characterization of Ui, and the uniqueness result recovers his.

The D-GBNE handles both incomplete information and shared constraints
simultaneously, which neither classical result does on its own.

---

## Summary of Assumptions

| Result | Assumptions Required |
|---|---|
| Existence of D-GBNE | 1, 2, 3 |
| Uniqueness of variational D-GBNE | 1, 2, 3, 4, 5, 6 |

---

## References

1. Arrow, K. J., and Debreu, G. *Existence of an equilibrium for a competitive economy.*  
   [https://doi.org/10.2307/1907353](https://doi.org/10.2307/1907353)

2. Kakutani, S. *A generalization of Brouwer's fixed point theorem.*  
   [https://doi.org/10.1215/S0012-7094-41-00838-4](https://doi.org/10.1215/S0012-7094-41-00838-4)

3. Rosen, J. B. *Existence and uniqueness of equilibrium points for concave N-person games.*  
   [https://doi.org/10.2307/1911749](https://doi.org/10.2307/1911749)

4. Ui, T. *Bayesian Nash equilibrium and variational inequalities.*  
   [https://doi.org/10.1016/j.jmateco.2016.01.003](https://doi.org/10.1016/j.jmateco.2016.01.003)