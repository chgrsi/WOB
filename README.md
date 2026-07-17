# WOB (Wasserstein Optimal Bidding)

WOB is a Julia-based optimization framework for day-ahead energy market bidding. It implements a data-driven Wasserstein DRO model to find the optimal energy nomination $n^*$ from the side of a Balance Responsible Party (BRP). By hedging against the uncertainty of renewable generation and the dual-pricing imbalance penalties, WOB seeks to limit extreme losses.

### Run the code

To run the experiments and generate out-of-sample results, you can use the following command line.

```
# Run experiments for a given epsilon
julia experiments_one_run.jl
```
By default, outputs are saved as CSV at results/one_run. 

You can then compile the results into PDF plots, using the following command line.
```
# Make plots
julia make_plots_one_run.jl
```
By default, plots are saved as PDF at plots/one_run.

### Play with the parameters

Depending on your prior as a decision-maker, you can change the level of robustness via the epsilon parameter as well as the modelization for the uncertainties. 
This can be done directly in ```experiments_one_run.jl```, for instance by uncommenting another modelization.
