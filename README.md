# WOB (Wasserstein Optimal Bidding)

WOB is a Julia-based optimization framework for day-ahead energy market bidding. It implements a data-driven Wasserstein DRO model to find the optimal energy nomination $n^*$ for a Balance Responsible Party (BRP). By hedging against the uncertainty of renewable generation and the punitive penalties of asymmetric dual-pricing imbalance markets, WOB seeks to limit extreme losses.

## Problem Formulation

We consider the expected profit maximization problem for a fixed delivery hour. 
Let $s$ denote the day-ahead clearing price, $g$ the actual (realized) generation, and $n$ the day-ahead nomination. 
The imbalance settlement is governed by the positive and negative imbalance prices, denoted $\text{PREP}$ (penalty for surplus) and $\text{PREN}$ (penalty for deficit), respectively. 
Under normal market conditions, we assume $\text{PREP} \le s \le \text{PREN}$. The ex-post profit function is given by:

$$\mathcal{R}(n, g, s, \text{PREP}, \text{PREN}) = s \cdot n + \text{PREP}\cdot (g - n)^+ - \text{PREN}\cdot (n - g)^+$$

To start with, we use generated data and do not allow negative prices. We proceed incrementally, in the following order.

- 1°) **Stochastic generation only** $\leadsto$ WOB_newsvendor.jl
- 2°) **Stochastic day-ahead prices** with PREP and PREN modeled as linear offsets of the day-ahead price, e.g., $\text{PREP} = s - \delta_{\text{surplus}}$ and $\text{PREN} = s + \delta_{\text{deficit}}$ $\leadsto$ WOB_stochastic_prices.jl
- 3°) **Stochastic imbalance prices** $\leadsto$ WOB_stochastic_imbal.jl

As a second step, we enable negative prices and replace generated data with real-world data.

## Quick Start

Clone the repository:
   ```bash
   git clone [https://github.com/chgrsi/WOB.git](https://github.com/chgrsi/WOB.git)
   cd WOB
