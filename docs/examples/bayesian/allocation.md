# Wireless Bandwidth Allocation Game

Three vehicles compete for wireless bandwidth from a roadside base station under incomplete information about each other's connection durations. Adapted from the Bayesian auction game of Akkarajitsakul et al. by replacing the proportional Kelly mechanism with an explicit shared interim constraint and recasting the problem as a D-GBNE.

**Players:** $N = 3$ vehicles, each controlling a type-indexed bandwidth request $a_i = (a_i^{\theta_i^1},\, a_i^{\theta_i^2}) \in \mathbb{R}^2$

## Types

| Player | Type set $\Theta_i$ | Probabilities $\eta_i$ |
|--------|---------------------|------------------------|
| 1 | $\{5.73,\; 10.64\}$ | $(0.3,\; 0.7)$ |
| 2 | $\{3.82,\; 7.09\}$ | $(0.7,\; 0.3)$ |
| 3 | $\{4.77,\; 8.86\}$ | $(0.7,\; 0.3)$ |

Types represent short and long connection durations (normalized by $\alpha = 9$). Higher type values indicate longer connections with greater marginal benefit from bandwidth. Types are drawn independently and each player's action set is $A_i(\theta_i) = [0, 1]$ for all types.

## Objective Functions

Each player $i$ maximizes the ex-ante expected utility $\Upsilon_i = \sum_{\theta_i} \eta_i(\theta_i)\,\rho_i(a_i^{\theta_i}, \theta_i)$, where the interim utility is:

$$\rho_i(a_i^{\theta_i}, \theta_i) = \theta_i \cdot \log\\!\left(1 + \gamma\, a_i^{\theta_i}\right) - \delta\, a_i^{\theta_i}$$

with $\gamma = 5$ and $\delta = 11$.

## Constraints

The shared capacity $C = 1$ is enforced in the **interim sense**: for each player $i$ and each type $\theta_i \in \Theta_i$,

$$h_i(a_i^{\theta_i}, a_{-i} \mid \theta_i) = a_i^{\theta_i} + \sum_{j \neq i} \sum_{\theta_j \in \Theta_j} \eta_j(\theta_j)\, a_j^{\theta_j} - C \leq 0$$

## Known Solution

The energy function method converges to $E^* \approx 1.0 \times 10^{-7}$. At equilibrium, short-connection types are **unconstrained** ($\lambda_i = 0$, $h_i < 0$) and play their interior optima, while long-connection types are **capacity-constrained** ($\lambda_i > 0$, $h_i \approx 0$), consistent with complementary slackness.

| Player $i$ | Type $\theta_i$ | $\eta_i(\theta_i)$ | $a^*_{i,\theta_i}$ | $\lambda^*_{i,\theta_i}$ | $h_i$ |
|:-----------:|:---------------:|:-------------------:|:-------------------:|:------------------------:|:------:|
| 1 | 5.73 | 0.3 | 0.3209 | 0 | −0.2372 |
| 1 | 10.64 | 0.7 | 0.5581 | 2.1242 | ≈ 0 |
| 2 | 3.82 | 0.7 | 0.1473 | 0 | −0.1017 |
| 2 | 7.09 | 0.3 | 0.2490 | 1.4377 | ≈ 0 |
| 3 | 4.77 | 0.7 | 0.2336 | 0 | −0.1016 |
| 3 | 8.86 | 0.3 | 0.3351 | 1.6670 | ≈ 0 |

## Implementation

```python
--8<-- "./problems/bayesian/AllocationGame_Akkarajitsakul.py"
```

## References

- **[Ho, 2026]** Andres Ho. *Discrete Generalized Bayesian Nash Equilibrium*. February 2026.
- **[Akkarajitsakul et al., 2011]** Khajonpong Akkarajitsakul, Ekram Hossain, and Dusit Niyato. Distributed resource allocation in wireless networks under uncertainty and application of Bayesian game. *IEEE Communications Magazine*, 49(8):120–127, 2011.
- **[Tsitsiklis & Xu, 2010]** John N. Tsitsiklis and Yunjian Xu. Bayesian proportional resource allocation games. In *Proc. 48th Allerton Conference on Communication, Control, and Computing*, pages 1556–1561. IEEE, 2010.