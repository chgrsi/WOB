# WOB (Wasserstein Optimal Bidding)

WOB is a Julia-based optimization framework for day-ahead energy market bidding. It implements a data-driven Wasserstein DRO model to find the optimal energy nomination $n^*$ by hedging against the uncertainty of renewable generation and asymmetric dual-pricing imbalance markets.

We consider the following minimization problem, noting $s$ the day-ahead price, $g$ the actual generation, and (PREP,PREN) the imbalance prices.

$$\mathcal{L}(n,g,s,r) = -ns+\text{PREN}(n-g)_+ - \text{PREP}(g-n)_+$$

To start with, we use generated data and do not allow negative prices. We proceed incrementally, in the following order.

- 1°) **Stochastic generation only (DRO-Newsvendor)** 
- 2°) **Stochastic prices** with PREP and PREN modeled as linear offsets of the day-ahead price, e.g., $\text{PREP} = s - \delta_{\text{surplus}}$ and $\text{PREN} = s + \delta_{\text{deficit}}$
- 3°) **Stochastic imbalance prices**

As a second step, we enable negative prices and replace generated data with real-world data.

## Quick Start

Clone the repository:
   ```bash
   git clone [https://github.com/charlesgarrisi/WOB.git](https://github.com/charlesgarrisi/WOB.git)
   cd WOB
