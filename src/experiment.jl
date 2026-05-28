
using JuMP
using HiGHS
using Distributions
using LinearAlgebra
using Statistics
using DataFrames
using CSV
using Random

include("run_experiments.jl")

Random.seed!(42)

###
# Config
###
GMM = (mu1 = -20.0, sig1 = 50.0, w1 = 0.5, mu2 = 1000.0, sig2 = 1.0, w2 = 0.5)
N_oos = 100_000
N_sims = 200
k_val = 1.2
Xi = (s_min=0.0, s_max = 500.0, d_min = -1000.0, d_max = 1000.0)
N_train = 3
W_mult = (g = 1.0, s = 1.0, d = 1.0)

###
# Probability distribution
###

law_s = LocationScale(90.0, 35.0, TDist(3))
law_d = MixtureModel([Normal(GMM.mu1, GMM.sig1), Normal(GMM.mu2, GMM.sig2)], [GMM.w1, GMM.w2])
law_f, law_zeta = truncated(Normal(0.7, 0.2), 0.0, 1.0), Normal(0.0, 0.15)

# Out of sample scenarios
g_oos = clamp.((1.0 .+ rand(law_zeta, N_oos)) .* rand(law_f, N_oos), 0.0, 1.0)
s_oos_b, d_oos_b = rand(law_s, N_oos), rand(law_d, N_oos)
s_oos = clamp.(s_oos_b, Xi.s_min, Xi.s_max)
d_oos = clamp.(d_oos_b, Xi.d_min, Xi.d_max)

# Solve robust formulation
n_ro = solve_ro(Xi)
ro_metrics = compute_oos_metrics(n_ro, g_oos, s_oos, d_oos)

# Training set
g_train = clamp.((1.0 .+ rand(law_zeta, N_train)) .* rand(law_f, N_train), 0.0, 1.0)
s_train = clamp.(rand(law_s, N_train), Xi.s_min, Xi.s_max)
d_train = clamp.(rand(law_d, N_train), Xi.d_min, Xi.d_max)

n_saa = solve_saa(g_train, s_train, d_train)
saa_metrics = compute_oos_metrics(n_saa, g_oos, s_oos, d_oos)

W = ones(5)
epsilons = 10 .^range(2, stop=4, length=100)

# Compare three different DROs formulations

n_dros_ex = [solve_dro_exact(g_train, s_train, d_train, Xi, W, eps) for eps in epsilons]
n_dros_mc = [solve_dro_approx_mccormick(g_train, s_train, d_train, Xi, W, eps) for eps in epsilons]
n_dros_box = [solve_dro_approx_box(g_train, s_train, d_train, Xi, W, eps) for eps in epsilons]
