# WOB (Wasserstein Optimal Bidding)

WOB is a Julia-based optimization framework for day-ahead energy market bidding. It implements a data-driven Wasserstein DRO model to find the optimal energy nomination $n^*$ for a Balance Responsible Party (BRP). By hedging against the uncertainty of renewable generation and the punitive penalties of asymmetric dual-pricing imbalance markets, WOB seeks to limit extreme losses.
