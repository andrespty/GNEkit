# Quadratic Utility Game

A Discrete Generalized Bayesian Nash Equilibrium (D-GBNE) with two players, each having two private types. Adapted from the GNEP literature by introducing discrete private types and replacing the standard shared constraint with an interim constraint that scales each player's resource consumption by their type.

**Players:** $N = 2$, each controlling a type-indexed action vector $a_i = (a_i^{\theta_i^1},\, a_i^{\theta_i^2}) \in \mathbb{R}^2$

## Types

| Player | Type set $\Theta_i$ | Probabilities $\eta_i$ |
|--------|---------------------|------------------------|
| 1 | $\{3,\; 7\}$ | $(0.8,\; 0.2)$ |
| 2 | $\{2,\; 8\}$ | $(0.4,\; 0.6)$ |

Types represent two demand levels. Types are drawn independently and each player's action set is $A_i(\theta_i) = [0, 1]$ for all types.

## Objective Functions

Each player $i$ maximizes the ex-ante expected utility $\Upsilon_i = \sum_{\theta_i} \eta_i(\theta_i)\,\rho_i(a_i^{\theta_i}, \theta_i)$, where the interim utilities are:

$$\rho_1(a_1^{\theta_1}, \theta_1) = a_1^{\theta_1} \cdot \theta_1 - \bigl(a_1^{\theta_1} - 1\bigr)^2$$

$$\rho_2(a_2^{\theta_2}, \theta_2) = a_2^{\theta_2} \cdot \theta_2 - \bigl(a_2^{\theta_2} - 0.5\bigr)^2$$

Neither player's utility depends directly on the opponent's action.

## Constraints

The shared capacity $C = 1$ is enforced in the **interim sense**: for each player $i$ and each type $\theta_i \in \Theta_i$,

$$h_i(a_i^{\theta_i}, a_{-i} \mid \theta_i) = a_i^{\theta_i} \cdot \theta_i + \sum_{j \neq i} \sum_{\theta_j \in \Theta_j} \eta_j(\theta_j)\, a_j^{\theta_j} - C \leq 0$$

## Known Solution

Since the interim utilities have no cross-player terms, all off-diagonal blocks of the Jacobian $J_\Upsilon(a)$ vanish. The diagonal entries are $-2\eta_i(\theta_i)$, giving

$$J_\Upsilon(a) = \mathrm{diag}(-1.6,\,-0.4,\,-0.8,\,-1.2),$$

which is negative definite — guaranteeing a unique variational equilibrium by Theorem 4.2.

Assuming active constraints and applying KKT stationarity, the unique equilibrium is:

| Player $i$ | Type $\theta_i$ | $\eta_i(\theta_i)$ | $a^*_{i,\theta_i}$ |
|:-----------:|:---------------:|:-------------------:|:-------------------:|
| 1 | 3 | 0.8 | 0.26301599 |
| 1 | 7 | 0.2 | 0.11273297 |
| 2 | 2 | 0.4 | 0.38356936 |
| 2 | 8 | 0.6 | 0.09588026 |

Higher types request strictly less resource, since the constraint penalizes their consumption more heavily per unit.

## Implementation

```python
--8<-- "./problems/bayesian/QuadraticGame.py"
```


## References

- **[Ho, 2026]** Andres Ho. *Discrete Generalized Bayesian Nash Equilibrium*. February 2026.
- **[Facchinei & Kanzow, 2010]** Francisco Facchinei and Christian Kanzow. Generalized Nash Equilibrium Problems. *Annals of Operations Research*, 175(1):177–211, 2010.
