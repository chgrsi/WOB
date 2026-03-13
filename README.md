# WOB (Wasserstein Optimal Bidding)

WOB is a Julia-based optimization framework for day-ahead energy market bidding. It implements a data-driven Wasserstein DRO model to find the optimal energy nomination $n^*$ for a Balance Responsible Party (BRP). By hedging against the uncertainty of renewable generation and the punitive penalties of asymmetric dual-pricing imbalance markets, WOB seeks to avoid dramatic losses.

## Problem Formulation

We consider the following minimization problem, noting $s$ the day-ahead price, $g$ the actual generation, and $n$ is the day-ahead nomination. The imbalance settlement is governed by th positive and negative imbalance prices ($\text{PREP}$ and $\text{PREN)}$ respectively).

$$\mathcal{L}(n,g,s,r) = -n\cdot s+\text{PREN}\cdot(n-g)^+ - \text{PREP}\cdot(g-n)^+$$

To start with, we use generated data and do not allow negative prices. We proceed incrementally, in the following order.

- 1°) **Stochastic generation only** $\leadsto$ DRO_newsvendor.jl
- 2°) **Stochastic day-ahead prices** with PREP and PREN modeled as linear offsets of the day-ahead price, e.g., $\text{PREP} = s - \delta_{\text{surplus}}$ and $\text{PREN} = s + \delta_{\text{deficit}}$ $\leadsto$ DRO_stochastic_prices.jl
- 3°) **Stochastic imbalance prices** modeled independently to capture asymmetric market dynamics

As a second step, we enable negative prices and replace generated data with real-world data.

## Quick Start

Clone the repository:
   ```bash
   git clone [https://github.com/chgrsi/WOB.git](https://github.com/chgrsi/WOB.git)
   cd WOB
