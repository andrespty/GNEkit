# Multistatic Radar Power Allocation Game

Two radars in a multistatic network selfishly choose transmission powers under private channel gain realizations, subject to a shared aggregate interference constraint. Adapted from the Bayesian SINR maximization game of Deligiannis & Lambotharan by replacing per-radar power budgets with a single coupled interference limit, motivated by radar–communication coexistence where regulatory limits bound the total interference a radar network may cause to a co-channel communication system.

**Players:** $N = 2$ radars, each controlling a type-indexed power allocation $a_i = (a_i^{\theta_i^1},\, a_i^{\theta_i^2}) \in \mathbb{R}^2$

## Types

| Player | Type set $\Theta_i$ | Probabilities $\eta_i$ |
|--------|---------------------|------------------------|
| 1 | $\{1.5,\; 4.0\}$ | $(0.4,\; 0.6)$ |
| 2 | $\{1.0,\; 3.0\}$ | $(0.7,\; 0.3)$ |

Types represent channel gain realizations. Higher types correspond to stronger channels with greater marginal SINR benefit per unit power. Types are drawn independently and each player's action set is $A_i(\theta_i) = [0, 1]$ for all types.

## Objective Functions

Each player $i$ maximizes the ex-ante expected utility $\Upsilon_i = \sum_{\theta_i} \eta_i(\theta_i)\,\rho_i(a_i^{\theta_i}, a_{-i}, \theta_i)$, where the interim SINR-based utility is:

$$\rho_i(a_i^{\theta_i}, a_{-i}, \theta_i) = \sum_{\theta_{-i}} \prod_{j \neq i} \eta_j(\theta_j) \cdot \frac{\theta_i\, a_i^{\theta_i}}{c_i\, a_i^{\theta_i} + \displaystyle\sum_{j \neq i} g_{ji}^{(\theta_j)}\, a_j^{\theta_j} + \sigma^2}$$

with $c_1 = 0.5$, $c_2 = 0.3$, $\sigma^2 = 0.1$, and cross-channel gains:

$$g_{12}^{(\theta_2)} = g_{21}^{(\theta_1)} = \begin{cases} 1.0 & \text{low type} \\ 4.0 & \text{high type} \end{cases}$$

Unlike the previous examples, **payoffs are coupled** across players through the interference terms in the denominator.

## Constraints

A shared aggregate interference limit $\bar{I} = 2.5$ is enforced in the **interim sense**: for each player $i$ and type $\theta_i$,

$$h_i(a_i^{\theta_i}, a_{-i} \mid \theta_i) = \tilde{g}_i^{(\theta_i)}\, a_i^{\theta_i} + \sum_{j \neq i} \sum_{\theta_j} \eta_j(\theta_j)\, \tilde{g}_j^{(\theta_j)}\, a_j^{\theta_j} - \bar{I} \leq 0$$

with interference footprint coefficients $\tilde{g}_1 = (1.0,\, 3.0)$ and $\tilde{g}_2 = (2.0,\, 5.0)$ (indexed by low/high type).

## Known Solution

The energy function method converges to $E^* \approx 4.5 \times 10^{-9}$. Radar 1 at its low type is the **only unconstrained** player-type pair ($\lambda \approx 0$, $h < 0$), because its interference coefficient $\tilde{g}_1^{(-)} = 1.0$ is the smallest among all pairs. All other pairs face a **binding constraint** ($\lambda > 0$, $h \approx 0$), with power reductions increasing with $\tilde{g}_i^{(\theta_i)}$.

| Player $i$ | Type $\theta_i$ | $\eta_i(\theta_i)$ | $a^*_{i,\theta_i}$ | $\lambda^*_{i,\theta_i}$ | $h_i$ |
|:-----------:|:---------------:|:-------------------:|:-------------------:|:------------------------:|:------:|
| 1 | 1.5 | 0.4 | 1.0000 | 0.0057 | ≈ 0 |
| 1 | 4.0 | 0.6 | 0.3333 | 0.6055 | ≈ 0 |
| 2 | 1.0 | 0.7 | 0.7499 | 0.1972 | ≈ 0 |
| 2 | 3.0 | 0.3 | 0.2999 | 0.1226 | ≈ 0 |

## Implementation

```python
--8<-- "./problems/bayesian/RadarPowerGame_Deligiannis.py"
```

## References

- **[Ho, 2026]** Andres Ho. *Discrete Generalized Bayesian Nash Equilibrium*. February 2026.
- **[Deligiannis & Lambotharan, 2017]** Anastasios Deligiannis and Sangarapillai Lambotharan. A Bayesian game theoretic framework for resource allocation in multistatic radar networks. In *Proc. IEEE Radar Conference (RadarConf)*, pages 546–551. IEEE, 2017.
- **[Shi et al., 2017]** Chenguang Shi, Sana Salous, Fei Wang, and Jianjiang Zhou. Power allocation for target detection in radar networks based on low probability of intercept: A cooperative game theoretical strategy. *Radio Science*, 52(8):1030–1045, 2017.
- **[Shi et al., 2018a]** Chenguang Shi, Fei Wang, Sana Salous, Jianjiang Zhou, and Zhentao Hu. Nash bargaining game-theoretic framework for power control in distributed multiple-radar architecture underlying wireless communication system. *Entropy*, 20(4):267, 2018.
- **[Shi et al., 2018b]** Chenguang Shi, Fei Wang, Mathini Sellathurai, and Jianjiang Zhou. Non-cooperative game theoretic power allocation strategy for distributed multiple-radar architecture in a spectrum sharing environment. *IEEE Access*, 6:17787–17800, 2018.