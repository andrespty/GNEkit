# Problem A9a

A GNEP arising from **power allocation in telecommunications**, described in detail by Pang, Scutari, Facchinei, and Wang (2008). This instance uses $K = 8$ channels.

**Players:** $N = 7$, each transmitting to a different Base Station over $K = 8$ channels.

## Optimization Problems

Each player $\nu$ minimizes total transmit power subject to a Quality-of-Service (QoS) constraint:

$$\min_{x^\nu}\; \sum_{i=1}^{K} x^\nu_i \qquad \text{subject to} \quad x^\nu \geq 0$$

$$\sum_{i=1}^{K} \log_2\\!\left(1 + \frac{h^{\nu\nu}_i\, x^\nu_i}{(\sigma^\nu_i)^2 + \displaystyle\sum_{\mu \neq \nu} h^{\nu\mu}_i\, x^\mu_i}\right) \geq L^\nu$$

with $\sigma^\nu_i = 0.3162$ for all $\nu, i$ and $L^\nu = 8$ for all players.

The $7 \times 56$ coefficient matrix $h^{\nu\mu}_i$ (channel gains) is given in full in Appendix A.9a of Facchinei and Kanzow (2009).

## Implementation

```python
--8<-- "./problems/gnep/ProblemA9a.py"
```
