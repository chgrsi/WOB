using Distributions
using DataFrames
using Random
using CSV
include("dro_lib.jl")

Random.seed!(2)

# Config
k_val = 1.2
tau = (k_val - 1.0) / (k_val + 1.0)
Xi = (s_min = -500.0, s_max = 4000.0, d_min = -14000.0, d_max = 10000.0)
N_oos = 100_000
N_train = 30
W_mult = (g = 1.0, s = 1.0, d = 1.0)
epsilons = 10 .^ range(-4, stop = 4, length = 50)

#############
# Modelization of the uncertainties 
#############
# Generation
# law_g = Uniform(0.0, 1.0) # agnostic
law_g = Normal(0.5, 0.15) # normal
# law_g = Beta(2.0, 5.0) # low solar
# law_g = Beta(5.0, 2.0) # high solar

# Spot prices
# law_s = Normal(40.0, 35.0) # low prices                                   
# law_s = Normal(90.0, 35.0) # normal
# law_s = Normal(-90.0, 100.0) # negative
# law_s = MixtureModel([Normal(90.0, 35.0), Normal(1000.0, 100.0), Normal(-90.0, 100.0)], [0.8, 0.1, 0.1]) # mixture
law_s = MixtureModel([Normal(0.0, 35.0), Normal(90.0, 25.0), Normal(90.0, 100.0)], [0.4, 0.4, 0.2]) # mixture
# law_s = LocationScale(60.0, 50.0, TDist(3)) # heavy tails

# Imbalance spread
# law_d = Normal(0.0, 150.0) # normal                                     
# law_d = Normal(0.0, 300.0) # wider
# law_d = MixtureModel([Normal(0.0, 50.0), Normal(1000.0, 100.0), Normal(-1000.0, 100.0)], [0.95, 0.025, 0.025]) # rare extreme imbalance
law_d = LocationScale(0.0, 130.0, TDist(2.5)) # heavy tails

# Out-of-sample scenarios + Training data
Random.seed!(1)
g_oos = clamp.(rand(law_g, N_oos), 0.0, 1.0)
g_train = clamp.(rand(law_g, N_train), 0.0, 1.0)
Random.seed!(2)
s_oos = clamp.(rand(law_s, N_oos), Xi.s_min, Xi.s_max)
s_train = clamp.(rand(law_s, N_train), Xi.s_min, Xi.s_max)
Random.seed!(3)
d_oos = clamp.(rand(law_d, N_oos), Xi.d_min, Xi.d_max)
d_train = clamp.(rand(law_d, N_train), Xi.d_min, Xi.d_max)
check_no_arbitrage(s_oos, d_oos, tau)

# Training data
W = scaling_weights(g_train, s_train, d_train; mult = W_mult)

# Benchmarks
n_ro_box = solve_ro_box(Xi, tau)
n_ro_mccormick = solve_ro_mccormick(Xi, tau)
n_saa = solve_saa(g_train, s_train, d_train, tau)
n_saa_exact = solve_saa_exact(g_train, s_train, d_train, tau)

ro_box_metrics = compute_oos_metrics(n_ro_box,  g_oos, s_oos, d_oos, tau)
ro_mccormick_metrics = compute_oos_metrics(n_ro_mccormick, g_oos, s_oos, d_oos, tau)
saa_metrics = compute_oos_metrics(n_saa, g_oos, s_oos, d_oos, tau)
saa_exact_metrics = compute_oos_metrics(n_saa_exact, g_oos, s_oos, d_oos, tau)

benchmark_df = DataFrame(
    Method = ["RO-box", "RO-mccormick", "SAA", "SAA exact"],
    n = round.([n_ro_box, n_ro_mccormick, n_saa, n_saa_exact], digits=3),
    mean = round.([ro_box_metrics.mean, ro_mccormick_metrics.mean, saa_metrics.mean, saa_exact_metrics.mean], digits=2),
    cvar = round.([ro_box_metrics.cvar, ro_mccormick_metrics.cvar, saa_metrics.cvar, saa_exact_metrics.cvar], digits=2),
    sharpe = round.([ro_box_metrics.sharpe, ro_mccormick_metrics.sharpe, saa_metrics.sharpe, saa_exact_metrics.sharpe], digits=3),
    sortino = round.([ro_box_metrics.sortino, ro_mccormick_metrics.sortino, saa_metrics.sortino, saa_exact_metrics.sortino], digits=3),
    neg_freq = round.([ro_box_metrics.neg_freq, ro_mccormick_metrics.neg_freq, saa_metrics.neg_freq, saa_exact_metrics.neg_freq], digits = 3),
    p_deficit = round.([ro_box_metrics.profit_deficit, ro_mccormick_metrics.profit_deficit, saa_metrics.profit_deficit, saa_exact_metrics.profit_deficit], digits = 2),
    p_surplus = round.([ro_box_metrics.profit_surplus, ro_mccormick_metrics.profit_surplus, saa_metrics.profit_surplus, saa_exact_metrics.profit_surplus], digits = 2),
)
println("\n##### Baseline models ####")
show(benchmark_df; allrows = true)
println("\n")

# Compare DRO formulations over a grid of epsilons
n_ex, is_ex, _ = solve_dro_exact_grid(g_train, s_train, d_train, Xi, W, tau, epsilons)
n_mc, is_mc, _ = solve_dro_approx_mccormick_grid(g_train, s_train, d_train, Xi, W, tau, epsilons)
n_bx, is_bx, _ = solve_dro_approx_box_grid(g_train, s_train, d_train, Xi, W, tau, epsilons)

# Check conservatism in-sample
println("max violation McCormick > exact: ", maximum(is_mc .- is_ex))
println("max violation box > McCormick:   ", maximum(is_bx .- is_mc))

# Report
oosm(nv) = [compute_oos_metrics(n, g_oos, s_oos, d_oos, tau) for n in nv]
m_ex, m_mc, m_bx = oosm(n_ex), oosm(n_mc), oosm(n_bx)
 
sel = unique(round.(Int, range(1, length(epsilons); length = 9)))

report_rows(method, decisions, metrics) = DataFrame(
    method = method,
    epsilon = round.(epsilons[sel], sigdigits = 3),
    n = round.(decisions[sel], digits = 3),
    p_mean = round.(getfield.(metrics[sel], :mean), digits = 2),
    cvar = round.(getfield.(metrics[sel], :cvar), digits = 2),
    sharpe = round.(getfield.(metrics[sel], :sharpe), digits = 3),
    sortino = round.(getfield.(metrics[sel], :sortino), digits = 3),
    neg_freq = round.(getfield.(metrics[sel], :neg_freq), digits = 3),
    p_deficit = round.(getfield.(metrics[sel], :profit_deficit), digits = 2),
    p_surplus = round.(getfield.(metrics[sel], :profit_surplus), digits = 2))

report = vcat(report_rows("DRO-exact", n_ex, m_ex),
              report_rows("DRO-approx-mccormick", n_mc, m_mc),
              report_rows("DRO-approx-box", n_bx, m_bx))

println("\n##### DRO models ####")
show(report; allrows = true)
println()

println("\nOOS profit by fixed n:")
for n in 0.0:0.1:1.0
    m = compute_oos_metrics(n, g_oos, s_oos, d_oos, tau)
    println("  n = ", n, "   mean = ", round(m.mean, digits = 2),
            "   cvar = ", round(m.cvar, digits = 1))
end

# Report the range of epsilons within 1% of the best mean 
means = getfield.(m_ex, :mean)
best = maximum(means)
good = epsilons[means .>= best - 0.01 * abs(best)]
println("\nbest mean = ", round(best, digits = 2),
        " for epsilon in [", round(minimum(good), sigdigits = 2),
        ", ", round(maximum(good), sigdigits = 2), "]")

# first epsilon where each method's decision moves away from SAA
for (name, decisions) in (("DRO-exact", n_ex), ("DRO-approx-mccormick", n_mc), ("DRO-approx-box", n_bx))
    j = findfirst(abs.(decisions .- decisions[1]) .> 0.01)
    println("first move ($name): ", j === nothing ? "never" : round(epsilons[j], sigdigits = 2))
end

mkpath("results/one_run")


metrics_rows(method, decisions, insample, metrics) = DataFrame(
    method = method, 
    epsilon = epsilons, 
    n = decisions, 
    insample_profit = insample,
    mean = getfield.(metrics, :mean), 
    cvar = getfield.(metrics, :cvar),
    sharpe = getfield.(metrics, :sharpe), 
    sortino = getfield.(metrics, :sortino),
    neg_freq = getfield.(metrics, :neg_freq),
    regret_mean = getfield.(metrics, :regret_mean), 
    regret_q95 = getfield.(metrics, :regret_q95),
    profit_deficit = getfield.(metrics, :profit_deficit), 
    profit_surplus = getfield.(metrics, :profit_surplus))

CSV.write("results/one_run/dro_metrics.csv",
          vcat(metrics_rows("DRO-exact", n_ex, is_ex, m_ex),
               metrics_rows("DRO-approx-mccormick", n_mc, is_mc, m_mc),
               metrics_rows("DRO-approx-box", n_bx, is_bx, m_bx)))

ref_row(method, n, m) = DataFrame(
        method = method, 
        n = n,
        mean = m.mean, 
        cvar = m.cvar, 
        sharpe = m.sharpe, 
        sortino = m.sortino,
        neg_freq = m.neg_freq,
        regret_mean = m.regret_mean, 
        regret_q95 = m.regret_q95,
        profit_deficit = m.profit_deficit, 
        profit_surplus = m.profit_surplus)

CSV.write("results/one_run/references.csv",
          vcat(ref_row("SAA", n_saa, saa_metrics),
               ref_row("SAA-exact", n_saa_exact, saa_exact_metrics),
               ref_row("RO-mccormick", n_ro_mccormick, ro_mccormick_metrics),
               ref_row("RO-box", n_ro_box, ro_box_metrics)))
