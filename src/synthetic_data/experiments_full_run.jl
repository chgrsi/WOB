using Distributions
using DataFrames
using CSV
using Statistics
using Random
include("dro_lib.jl")

# Config
N_train = 3
N_sims = 300
N_oos = 100_000
epsilons = 10 .^ range(-4, stop = 4, length = 30)
k_val = 1.2
tau = (k_val - 1.0) / (k_val + 1.0)
W_mult = (g = 1.0, s = 1.0, d = 1.0)
ngrid = 101
scenarios = [(name = "pos", s_min = 0.0), (name = "neg", s_min = -500.0)]
s_max, d_min, d_max = 4000.0, -14000.0, 10000.0

# Modelization of the uncertainties 
law_g = Normal(0.5, 0.15)
law_s = MixtureModel([Normal(90.0, 35.0), Normal(1000.0, 100.0), Normal(-90.0, 100.0)], [0.8, 0.1, 0.1]) # mixture
law_d = MixtureModel([Normal(0.0, 50.0), Normal(1000.0, 100.0), Normal(-1000.0, 100.0)], [0.95, 0.025, 0.025]) # rare extreme imbalance

Random.seed!(42)
g_oos = clamp.(rand(law_g, N_oos), 0.0, 1.0)
s_oos_base = rand(law_s, N_oos)
d_oos_base = rand(law_d, N_oos)

println("\n>>> Running $N_sims sims, N_train = $N_train, $(length(epsilons)) epsilons")
rows = NamedTuple[]

agg(metrics, f) = mean(getfield.(metrics, f))
nanagg(metrics, f) = (v = filter(!isnan, getfield.(metrics, f)); isempty(v) ? NaN : mean(v))

function summary_row(scen_name, method, decisions, metrics; insample = nothing, eps = NaN, lam = NaN)
    q(v, p) = quantile(v, p)
    oos_mean = getfield.(metrics, :mean)
    cvars = getfield.(metrics, :cvar)
    sharpes = getfield.(metrics, :sharpe)
    sortinos = getfield.(metrics, :sortino)
    negfreqs = getfield.(metrics, :neg_freq)
    return (Scenario = scen_name, Epsilon = eps, Method = method,
            N_opt = mean(decisions),
            N_Q20 = q(decisions, 0.2),
            N_Q80 = q(decisions, 0.8),
            Reliability = insample === nothing ? NaN : mean(oos_mean .>= insample),
            Profit_Mean = mean(oos_mean),
            Profit_Q20 = q(oos_mean, 0.2),
            Profit_Q80 = q(oos_mean, 0.8),
            CVaR5 = mean(cvars),
            CVaR5_Q20 = q(cvars, 0.2),
            CVaR5_Q80 = q(cvars, 0.8),
            Sharpe = mean(sharpes),
            Sharpe_Q20 = q(sharpes, 0.2),
            Sharpe_Q80 = q(sharpes, 0.8),
            Sortino = mean(sortinos),
            Sortino_Q20 = q(sortinos, 0.2),
            Sortino_Q80 = q(sortinos, 0.8),
            NegFreq = mean(negfreqs),
            NegFreq_Q20 = q(negfreqs, 0.2),
            NegFreq_Q80 = q(negfreqs, 0.8),
            Regret = agg(metrics, :regret_mean),
            ProfitDeficit = nanagg(metrics, :profit_deficit),
            ProfitSurplus = nanagg(metrics, :profit_surplus),
            Lambda = lam)
end

for scen in scenarios
    println("\n  -> Scenario: $(scen.name)")
    Xi = (s_min = scen.s_min, s_max = s_max, d_min = d_min, d_max = d_max)
    s_oos = clamp.(s_oos_base, Xi.s_min, Xi.s_max)
    d_oos = clamp.(d_oos_base, Xi.d_min, Xi.d_max)
    check_no_arbitrage(s_oos, d_oos, tau)

    # clairvoyant profit does not depend on n so we compute it once per scenario
    clair = max.(profit.(0.0, g_oos, s_oos, d_oos, tau),
                profit.(g_oos, g_oos, s_oos, d_oos, tau),
                profit.(1.0, g_oos, s_oos, d_oos, tau))
    get_metrics(n) = compute_oos_metrics(n, g_oos, s_oos, d_oos, tau; clairvoyant = clair)

    # deterministic benchmarks (support only)
    n_ro = solve_ro_mccormick(Xi, tau)
    n_ro_bx = solve_ro_box(Xi, tau)
    ro_m = get_metrics(n_ro)
    ro_bx_m = get_metrics(n_ro_bx)

    n_eps = length(epsilons)
    n_saa, n_saax = zeros(N_sims), zeros(N_sims)
    n_ex, is_ex, lam_ex = zeros(N_sims, n_eps), zeros(N_sims, n_eps), zeros(N_sims, n_eps)
    n_mc, is_mc, lam_mc = zeros(N_sims, n_eps), zeros(N_sims, n_eps), zeros(N_sims, n_eps)
    n_bx, is_bx, lam_bx = zeros(N_sims, n_eps), zeros(N_sims, n_eps), zeros(N_sims, n_eps)

    for sim in 1:N_sims
        Random.seed!(sim)
        g_tr = clamp.(rand(law_g, N_train), 0.0, 1.0)
        s_tr = clamp.(rand(law_s, N_train), Xi.s_min, Xi.s_max)
        d_tr = clamp.(rand(law_d, N_train), Xi.d_min, Xi.d_max)
        W = scaling_weights(g_tr, s_tr, d_tr; mult = W_mult)

        n_saa[sim] = solve_saa(g_tr, s_tr, d_tr, tau)
        n_saax[sim] = solve_saa_exact(g_tr, s_tr, d_tr, tau)

        n_ex[sim, :], is_ex[sim, :], lam_ex[sim, :] = solve_dro_exact_grid(g_tr, s_tr, d_tr, Xi, W, tau, epsilons; ngrid = ngrid)
        n_mc[sim, :], is_mc[sim, :], lam_mc[sim, :] = solve_dro_approx_mccormick_grid(g_tr, s_tr, d_tr, Xi, W, tau, epsilons)
        n_bx[sim, :], is_bx[sim, :], lam_bx[sim, :] = solve_dro_approx_box_grid(g_tr, s_tr, d_tr, Xi, W, tau, epsilons)

        sim % 20 == 0 && println("    sim $sim / $N_sims")
    end

    # conservatism chain (in-sample): must be <= 0 up to solver tolerance
    println("max violation McCormick > exact: ", maximum(is_mc .- is_ex))
    println("max violation box > McCormick:   ", maximum(is_bx .- is_mc))

    # benchmark metrics, aggregated over simulations
    saa_metrics = [get_metrics(n_saa[sim]) for sim in 1:N_sims]
    saax_metrics = [get_metrics(n_saax[sim]) for sim in 1:N_sims]
    push!(rows, summary_row(scen.name, "SAA", n_saa, saa_metrics))
    push!(rows, summary_row(scen.name, "SAA-exact", n_saax, saax_metrics))
    push!(rows, summary_row(scen.name, "RO-mccormick", [n_ro], [ro_m]))
    push!(rows, summary_row(scen.name, "RO-box", [n_ro_bx], [ro_bx_m]))

    # evaluate all sims out-of-sample for the DRO methods
    for (j, eps) in enumerate(epsilons)
        for (method, n_m, is_m, lam_m) in (("DRO-exact", n_ex, is_ex, lam_ex),
                                           ("DRO-approx-mccormick", n_mc, is_mc, lam_mc),
                                           ("DRO-approx-box", n_bx, is_bx, lam_bx))
            metrics = [get_metrics(n_m[sim, j]) for sim in 1:N_sims]
            push!(rows, summary_row(scen.name, method, n_m[:, j], metrics;
                                    insample = is_m[:, j], eps = eps,
                                    lam = mean(lam_m[:, j])))
        end
    end
end

mkpath("results/full_run")
out = "results/full_run/dro_metrics_N_$(N_train).csv"
CSV.write(out, DataFrame(rows))

