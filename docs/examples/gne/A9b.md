# Problem A9b

A larger instance of the telecommunications power allocation GNEP from Problem A9a, using $K = 16$ channels.

**Players:** $N = 7$, each transmitting to a different Base Station over $K = 16$ channels.

## Optimization Problems

Same formulation as Problem A9a:

$$\min_{x^\nu}\; \sum_{i=1}^{K} x^\nu_i \qquad \text{subject to} \quad x^\nu \geq 0$$

$$\sum_{i=1}^{K} \log_2\\!\left(1 + \frac{h^{\nu\nu}_i\, x^\nu_i}{(\sigma^\nu_i)^2 + \displaystyle\sum_{\mu \neq \nu} h^{\nu\mu}_i\, x^\mu_i}\right) \geq L^\nu$$

with $\sigma^\nu_i = 0.3162$ for all $\nu, i$ and $L^\nu = 16$ for all players.

The $7 \times 112$ coefficient matrix $h^{\nu\mu}_i$ is given in full in Appendix A.9b of Facchinei and Kanzow (2009).

## Implementation

```python
--8<-- "./problems/gnep/ProblemA9b.py"
```
