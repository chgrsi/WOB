# WOB (Wasserstein Optimal Bidding)

WOB is a Julia-based optimization framework for day-ahead energy market bidding. It implements a data-driven Wasserstein DRO model to find the optimal energy nomination $n^*$ from the side of a Balance Responsible Party (BRP). By hedging against the uncertainty of renewable generation and the dual-pricing imbalance penalties, WOB seeks to limit extreme losses.

### Run the code

To run the experiments and generate OOS metrics, you can either run all training sizes (3, 30, 300) or target a specific one (e.g., 30).

```
# Run all (3, 30, 300)
julia run_experiments.jl

# Run specific sample size
julia run_experiments.jl 30
```
By default, outputs are saved as CSV to results/raw/. 

You can then compile the results into PDF plots, using the following command line.
```
# Make plots
julia make_plots.jl
```
By default, plots are saved at plots/raw/.

### Sensitivity analysis

Modify the CONFIG in run_experiments.jl to test specific hypotheses, e.g.,
- high variance: ```sig1 = 100.0, sig2 = 200.0```
- fat tails: ```w1 = 0.8, w2 = 0.2```
- wide bounds: ```s_max = 2000.0, d_min = -5000.0, d_max = 5000.0```
- tight bounds: ```s_max = 200.0, d_min = -200.0, d_max = 200.0```
- asymmetric scaling: ```W_mult = (g = 10.0, s = 1.0, d = 1.0)```
- uniform scaling (cheaper peanalty): ```W_mult = (g = 0.1, s = 0.1, d = 0.1)```



