# Assumptions and Theoretical Results (GNEP)

We summarize the main theoretical results about GNEPs: when equilibria exist,
when they are unique, and what additional structure is needed to say more. Proofs are
omitted; references are provided for each result.

---

## Existence

The first question is whether a GNE exists at all. The answer depends on regularity
conditions on the players' objectives and feasible sets.

The foundational existence result goes back to [Debreu (1952)](#references), who proved existence
using a fixed-point argument. The idea is straightforward: define the **best response
map** of player $i$ as

$$
S_i(x_{-i}) = \arg\min_{x_i \in X_i(x_{-i})} f_i(x_i, x_{-i})
$$

and the joint best response map $S(x) = S_1(x_{-1}) \times \cdots \times S_N(x_{-N})$.
A GNE is exactly a fixed point of $S$. Existence then follows from Kakutani's
fixed-point theorem under the following conditions:

- Each $X_i(x_{-i})$ is nonempty, convex, and compact for all $x_{-i}$
- The map $x_{-i} \mapsto X_i(x_{-i})$ is continuous (in the set-valued sense)
- Each $f_i(\cdot, x_{-i})$ is convex and continuous

These are sufficient conditions, not necessary ones. The continuity assumption on the
constraint map — formally, that $X_i(x_{-i})$ is both upper and lower semicontinuous
in $x_{-i}$ — is the most delicate to verify in practice, and much of the more recent
existence literature is concerned with relaxing it.

---

## Uniqueness

<!-- Uniqueness of GNEs is the exception, not the rule. -->

When players share constraints, active shared constraints generically produce a
**continuum of equilibria**, not an isolated one. This is a fundamental difference
from classical NEPs, where isolated equilibria are the norm.

The standard uniqueness result is due to [Rosen (1965)](#references), and applies to the jointly
convex setting. Rosen showed that under a condition called **diagonal strict
convexity (DSC)** — a joint convexity condition on the weighted sum of players'
objectives — the **variational equilibrium** is unique. This does not mean the
GNEP has a unique GNE; it means that among all GNEs, there is exactly one that
also solves the associated variational inequality.

Outside the jointly convex setting, global uniqueness results are rare. Local
uniqueness of normalized equilibria has been studied under nondegeneracy conditions
(analogues of LICQ, strict complementarity, and second-order sufficiency at the game
level), and it has been shown that nondegeneracy holds generically.

---

## The Variational Equilibrium

Because GNEPs typically admit many equilibria, it is necessary to select among them.
The most widely used refinement is the **variational equilibrium (VE)**, also called
the normalized equilibrium.

In the jointly convex setting, where all players share the same constraint $g(x) \leq 0$,
the VE is defined as a GNE whose Lagrange multipliers for the shared constraint are
**identical across all players**. This symmetry condition — that every player assigns
the same price to the shared resource — has a natural economic interpretation and
significantly narrows the solution set.

Formally, the VE is a solution to the variational inequality $\text{VI}(X, F)$: find $x^* \in X$ such that

$$
F(x^*)^\top (x-x^*) \geq 0 \quad \forall x\in X
$$

where $X = \{x : g(x) \leq 0\}$ is the joint feasible set and
$F(x) = (\nabla_{x_i} f_i)_{i=1}^N$ is the concatenation of players' gradients.
Every VE is a GNE, but not every GNE is a VE.

The VE has become the de facto solution concept in computational work, largely because
it can be computed by solving a single variational inequality or a KKT system, rather
than a quasi-variational inequality.

---

## Summary of Assumptions

| Result | Key Assumptions |
|---|---|
| Existence (general GNEP) | Convexity and continuity of $f_i$; continuity of $X_i(\cdot)$ |
| Existence (jointly convex) | Convexity of $f_i$ and $g$; Slater's condition |
| Uniqueness of VE | Jointly convex + diagonal strict convexity (Rosen) |
| Local uniqueness of normalized NE | Nondegeneracy (GNEP-level LICQ + strict complementarity) |

---

## References

1. Debreu, G. *A social equilibrium existence theorem.*  
   [https://doi.org/10.1073/pnas.38.10.886](https://doi.org/10.1073/pnas.38.10.886)

2. Rosen, J. B. *Existence and uniqueness of equilibrium points for concave N-person games.*  
   [https://doi.org/10.2307/1911749](https://doi.org/10.2307/1911749)

3. Facchinei, F., and Kanzow, C. *Generalized Nash equilibrium problems.*  
   [https://doi.org/10.1007/s10107-006-0034-9](https://doi.org/10.1007/s10107-006-0034-9)

4. Izmailov, A. F., and Solodov, M. V. *On local uniqueness of normalized Nash equilibria.*  
   [https://doi.org/10.48550/arXiv.2205.13878](https://doi.org/10.48550/arXiv.2205.13878)

5. Kulkarni, A. A., and Shanbhag, U. V. *On the variational equilibrium as a refinement
   of the generalized Nash equilibrium.*  
   [https://doi.org/10.1016/j.automatica.2011.09.042](https://doi.org/10.1016/j.automatica.2011.09.042)