# WOB (Wasserstein Optimal Bidding)

WOB is a Julia-based optimization framework for day-ahead energy market bidding. It implements a data-driven Wasserstein DRO model to find the optimal energy nomination $n^*$ from the side of a Balance Responsible Party (BRP). By hedging against the uncertainty of renewable generation and the dual-pricing imbalance penalties, WOB seeks to limit extreme losses.
