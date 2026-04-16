# Discrete Generalized Bayesian Nash Equilibrium

## What is a D-GBNE?

A **Discrete Generalized Bayesian Nash Equilibrium (D-GBNE)** combines two
extensions of the classical Nash game:

- **Incomplete information**: each player has private information, called their
  **type**, drawn from a finite set. Players do not observe each other's types.
- **Coupled constraints**: the feasible actions of each player depend on the
  actions of others, as in a GNEP.

Neither framework alone is sufficient for many practical settings. The classical
Bayesian Nash Equilibrium (BNE) handles private information but assumes
uncoupled feasible sets. The GNEP handles coupled constraints but assumes
complete information. The D-GBNE handles both simultaneously.

---

## Formal Setup

Consider a game with $N$ players. Each player $i$ has a **private type**
$\theta_i$ drawn from a finite set $\Theta_i = \{\theta_i^1, \dots, \theta_i^{m_i}\}$.
The **joint type space** is $\Theta = \prod_{i=1}^N \Theta_i$.

Players' beliefs are governed by a joint distribution $\eta(\theta)$ over $\Theta$.
Given their own type $\theta_i$, player $i$ holds a conditional belief over
opponents' types:

$$
\eta_i(\theta_{-i} \mid \theta_i) = \frac{\eta(\theta)}{\eta_i(\theta_i)}
$$

Each player $i$ chooses an action $a_i$ from a type-dependent action set
$A_i(\theta_i) \subseteq \mathbb{R}^{d_i}$. A **pure strategy** is a mapping
$s_i : \Theta_i \to \bigcup_{\theta_i} A_i(\theta_i)$ such that $s_i(\theta_i) \in A_i(\theta_i)$
for all $\theta_i$.

---

## Objectives and Constraints

Because types are private, players evaluate their objectives and constraints
**in expectation over opponents' types**. For a fixed strategy profile
$s = (s_1, \dots, s_N)$, the **interim expected utility** and **interim expected
constraint** for player $i$ given type $\theta_i$ are:

$$
\rho_i(s_i, s_{-i} \mid \theta_i) = \sum_{\theta_{-i} \in \Theta_{-i}}
u_i(s_i(\theta_i),\, s_{-i}(\theta_{-i}),\, \theta_i,\, \theta_{-i})
\cdot \eta_i(\theta_{-i} \mid \theta_i)
$$

$$
h_i(s_i, s_{-i} \mid \theta_i) = \sum_{\theta_{-i} \in \Theta_{-i}}
g_i(s_i(\theta_i),\, s_{-i}(\theta_{-i}),\, \theta_i,\, \theta_{-i})
\cdot \eta_i(\theta_{-i} \mid \theta_i)
$$

The **interim feasible set** for player $i$ at type $\theta_i$ is:

$$
X_i(s_{-i}, \theta_i) = \bigl\{ a_i \in A_i(\theta_i) : h_i(a_i, s_{-i} \mid \theta_i) \leq 0 \bigr\}
$$

This set depends on the strategies of other players, which is what distinguishes
the D-GBNE from a standard Bayesian Nash game. Feasibility is imposed in the
**interim sense**: conditional on player $i$'s own type, not almost surely across
all type realizations.

---

## Equilibrium Definition

A strategy profile $s^* = (s_1^*, \dots, s_N^*)$ is a **D-GBNE** if, for every
player $i \in \mathcal{N}$ and every type $\theta_i \in \Theta_i$, the strategy
$s_i^*(\theta_i)$ solves:

$$
\max_{a_i \in A_i(\theta_i)} \rho_i(a_i, s_{-i}^* \mid \theta_i)
\qquad \text{subject to} \qquad h_i(a_i, s_{-i}^* \mid \theta_i) \leq 0
$$

Each player is best-responding to the other players' strategies, in expectation
over types they do not observe, while remaining feasible under the coupled
constraint — also evaluated in expectation.

---

## Reformulation as a GNEP

The key result of the framework is that a D-GBNE can be exactly reformulated
as a finite-dimensional GNEP. The idea is to extend each player's strategy into
a single action vector that collects their decisions across all type realizations:

$$
a_i = \bigl(a_i^{\theta_i^1},\, a_i^{\theta_i^2},\, \dots,\, a_i^{\theta_i^{m_i}}\bigr)
\in \mathcal{A}_i = \prod_{\theta_i \in \Theta_i} A_i(\theta_i)
$$

With this representation, the **ex-ante expected utility** and **lifted feasible
set** for player $i$ are:

$$
\Upsilon_i(a_i, a_{-i}) = \sum_{\theta_i \in \Theta_i} \eta_i(\theta_i) \cdot \rho_i(a_i^{\theta_i}, a_{-i} \mid \theta_i)
$$

$$
\mathcal{K}_i(a_{-i}) = \prod_{\theta_i \in \Theta_i} X_i(a_{-i}, \theta_i)
$$

The lifted feasible set $\mathcal{K}_i(a_{-i})$ enforces the interim constraint for
**every** type realization of player $i$. Each player then solves:

$$
\max_{a_i \in \mathcal{A}_i} \Upsilon_i(a_i, a_{-i})
\qquad \text{subject to} \qquad a_i \in \mathcal{K}_i(a_{-i})
$$

This is exactly a GNEP in the lifted action space. The equivalence is exact:
$s^*$ is a D-GBNE if and only if the corresponding lifted profile $a^*$ is a GNE
of this problem. Since each $\Theta_i$ is finite, the resulting GNEP is
finite-dimensional, enabling direct use of standard GNEP solvers.

---

## Relation to Existing Models

| Model | Private Types | Coupled Constraints |
|---|:---:|:---:|
| Nash Equilibrium Problem (NEP) | ✗ | ✗ |
| Bayesian Nash Equilibrium (BNE) | ✓ | ✗ |
| Generalized Nash Equilibrium Problem (GNEP) | ✗ | ✓ |
| **D-GBNE** | **✓** | **✓** |

The D-GBNE reduces to a standard BNE when constraints are decoupled
($\mathcal{K}_i(a_{-i}) = A_i$ for all $a_{-i}$), and reduces to a GNEP when there
is a single type per player ($m_i = 1$ for all $i$).

---

## References

1. Harsanyi, J. C. *Games with incomplete information played by Bayesian players.*  
   [https://doi.org/10.1287/mnsc.14.3.159](https://doi.org/10.1287/mnsc.14.3.159)

2. Facchinei, F., and Kanzow, C. *Generalized Nash equilibrium problems.*  
   [https://doi.org/10.1007/s10107-006-0034-9](https://doi.org/10.1007/s10107-006-0034-9)

3. Rosen, J. B. *Existence and uniqueness of equilibrium points for concave N-person games.*  
   [https://doi.org/10.2307/1911749](https://doi.org/10.2307/1911749)

4. Ui, T. *Bayesian Nash equilibrium and variational inequalities.*  
   [https://doi.org/10.1016/j.jmateco.2016.01.003](https://doi.org/10.1016/j.jmateco.2016.01.003)