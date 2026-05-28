using JuMP
using Gurobi
using HiGHS
using Distributions
using LinearAlgebra
using Statistics
using DataFrames
using CSV
using Random

const GRB_ENV = Gurobi.Env()

const TARGET_N_TRAINS = length(ARGS) > 0 ? [parse(Int, arg) for arg in ARGS] : [3, 30, 300]

const CONFIG = (
    N_sims = 200,
    N_oos = 100_000,
    epsilons = 10 .^ range(-4, stop=4, length=100),
    k_val = 1.2,
    Xi = (s_min = 0.0, s_max = 500.0, d_min = -1000.0, d_max = 1000.0),
    GMM = (mu1 = -20.0, sig1 = 50.0, w1 = 0.95, mu2 = 1000.0, sig2 = 1.0, w2 = 0.05),
    W_mult = (g = 1.0, s = 1.0, d = 1.0)
)
const tau_val = (CONFIG.k_val - 1.0) / (CONFIG.k_val + 1.0)

function compute_oos_metrics(n, g, s, delta)
    cp, cm = (s .+ delta) .* tau_val .- delta, (s .+ delta) .* tau_val .+ delta
    profits = sort!((g .* s) .- cp .* max.(g .- n, 0.0) .- cm .* max.(n .- g, 0.0))
    idx_5 = max(1, round(Int, 0.05 * length(profits)))
    return (mean = mean(profits), cvar5 = mean(view(profits, 1:idx_5)), profits=profits)
end

function product_bounds(xL, xU, yL, yU)
    vals = (xL*yL, xL*yU, xU*yL, xU*yU)
    return minimum(vals), maximum(vals)
end

function build_mccormick_support(Xi)
    s_min, s_max, d_min, d_max = Xi.s_min, Xi.s_max, Xi.d_min, Xi.d_max
    C = [
         1.0    0.0   0.0   0.0   0.0;
        -1.0    0.0   0.0   0.0   0.0;
         0.0    1.0   0.0   0.0   0.0;
         0.0   -1.0   0.0   0.0   0.0;
         0.0    0.0   1.0   0.0   0.0;
         0.0    0.0  -1.0   0.0   0.0;
         s_min  0.0   0.0  -1.0   0.0;
         s_max  1.0   0.0  -1.0   0.0;
        -s_min -1.0   0.0   1.0   0.0;
        -s_max  0.0   0.0   1.0   0.0;
         d_min  0.0   0.0   0.0  -1.0;
         d_max  0.0   1.0   0.0  -1.0;
        -d_min  0.0  -1.0   0.0   1.0;
        -d_max  0.0   0.0   0.0   1.0
    ]
    d = [1.0, 0.0, s_max, -s_min, d_max, -d_min, 0.0, s_max, -s_min, 0.0, 0.0, d_max, -d_min, 0.0]
    return C, d
end

function build_box_support(Xi)
    s_min = Xi.s_min
    s_max = Xi.s_max
    d_min = Xi.d_min
    d_max = Xi.d_max
    gL, gU = 0.0, 1.0
    sL, sU = s_min, s_max
    dL, dU = d_min, d_max

    zsL, zsU = product_bounds(gL, gU, sL, sU)
    zdL, zdU = product_bounds(gL, gU, dL, dU)

    lower = [gL, sL, dL, zsL, zdL]
    upper = [gU, sU, dU, zsU, zdU]

    dim = 5
    C = zeros(2dim, dim)
    rhs = zeros(2dim)

    for j in 1:dim
        # xi_j <= upper_j
        C[2j - 1, j] = 1.0
        rhs[2j - 1] = upper[j]

        # -xi_j <= -lower_j
        C[2j, j] = -1.0
        rhs[2j] = -lower[j]
    end

    return C, rhs, lower, upper
end

function solve_saa(g_train, s_train, d_train)
    m = Model(HiGHS.Optimizer); set_silent(m)
    N = length(g_train)
    @variables(m, begin
        0 <= n <= 1
        loss[1:N]
    end)
    for i in 1:N
        cp, cm = (s_train[i]+d_train[i])*tau_val - d_train[i], (s_train[i]+d_train[i])*tau_val + d_train[i]
        @constraint(m, loss[i] >= -g_train[i]*s_train[i] + cp*(g_train[i] - n))
        @constraint(m, loss[i] >= -g_train[i]*s_train[i] + cm*(n - g_train[i]))
    end
    @objective(m, Min, sum(loss)/N)
    optimize!(m)
    return value(n)
end

function solve_ro(Xi)
    m = Model(HiGHS.Optimizer); set_silent(m)
    C, d = build_mccormick_support(Xi)
    @variables(m, begin
        0 <= n <= 1
        wc
        g1[1:14] >= 0
        g2[1:14] >= 0
    end)

    B1 = [0.0, -tau_val*n, (1.0-tau_val)*n, tau_val-1.0, tau_val-1.0]
    B2 = [0.0, tau_val*n, (1.0+tau_val)*n, -(1.0+tau_val), -(1.0+tau_val)]

    @constraint(m, C'*g1 .== B1)
    @constraint(m, dot(d, g1) <= wc)
    @constraint(m, C'*g2 .== B2)
    @constraint(m, dot(d, g2) <= wc)
    @objective(m, Min, wc)
    optimize!(m)
    return value(n)
end

function solve_dro_exact(g_train, s_train, d_train, Xi, W, eps)
    N = length(g_train)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variables(m, begin
        0 <= n <= 1
        lam >= 0
        epi[1:N]
    end)
    for i in 1:N
        for g_k in [0.0, g_train[i], 1.0], s_k in [Xi.s_min, s_train[i], Xi.s_max], d_k in [Xi.d_min, d_train[i], Xi.d_max]
            dist = W[1]*abs(g_k-g_train[i]) + W[2]*abs(s_k-s_train[i]) + W[3]*abs(d_k-d_train[i])
            cp, cm = (s_k+d_k)*tau_val - d_k, (s_k+d_k)*tau_val + d_k
            @constraint(m, epi[i] >= -g_k*s_k + cp*(g_k - n) - lam*dist)
            @constraint(m, epi[i] >= -g_k*s_k + cm*(n - g_k) - lam*dist)
        end
    end

    n_opts, prof_is, lam_opts = Float64[], Float64[], Float64[]

    @objective(m, Min, lam*eps + sum(epi)/N)
    optimize!(m)
    return value.(n)
end

function solve_dro_approx_mccormick(g_train, s_train, d_train, Xi, W, eps)
    N = length(g_train)
    m = Model(HiGHS.Optimizer); set_silent(m)
    C, d = build_mccormick_support(Xi)
    @variables(m, begin
        0 <= n <= 1
        lam >= 0
        epi[1:N]
        g1[1:N, 1:14] >= 0
        g2[1:N, 1:14] >= 0
    end)

    B1 = [0.0, -tau_val*n, (1.0-tau_val)*n, tau_val-1.0, tau_val-1.0]
    B2 = [0.0, tau_val*n, (1.0+tau_val)*n, -(1.0+tau_val), -(1.0+tau_val)]

    for i in 1:N
        slack = d .- C * [g_train[i], s_train[i], d_train[i], g_train[i]*s_train[i], g_train[i]*d_train[i]]
        cp, cm = (s_train[i]+d_train[i])*tau_val - d_train[i], (s_train[i]+d_train[i])*tau_val + d_train[i]
        @constraint(m, -g_train[i]*s_train[i] + cp*(g_train[i]-n) + dot(g1[i,:], slack) <= epi[i])
        @constraint(m, C'*g1[i,:] .- B1 .<= lam .* W)
        @constraint(m, -(C'*g1[i,:] .- B1) .<= lam .* W)
        @constraint(m, -g_train[i]*s_train[i] + cm*(n-g_train[i]) + dot(g2[i,:], slack) <= epi[i])
        @constraint(m, C'*g2[i,:] .- B2 .<= lam .* W)
        @constraint(m, -(C'*g2[i,:] .- B2) .<= lam .* W)
    end
    n_opts, prof_is, lam_opts = Float64[], Float64[], Float64[]
    @objective(m, Min, lam*eps + sum(epi)/N)
    optimize!(m)
    return value(n)
end

function solve_dro_approx_box(g_train, s_train, d_train, Xi, W, eps)
    N = length(g_train)
    C, rhs, _, _ = build_box_support(Xi)
    K = size(C, 1)

    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variables(m, begin
        0 <= n <= 1
        lam >= 0
        epi[1:N]
        dual1[1:N, 1:K] >= 0
        dual2[1:N, 1:K] >= 0
    end)

    B1 = [0.0, -tau_val*n, (1.0-tau_val)*n, tau_val-1.0, tau_val-1.0]
    B2 = [0.0,  tau_val*n, (1.0+tau_val)*n, -(1.0+tau_val), -(1.0+tau_val)]

    for i in 1:N
        ξi = [g_train[i], s_train[i], d_train[i], g_train[i] * s_train[i], g_train[i] * d_train[i]]
        slack = rhs .- C * ξi

        @constraint(m, dot(B1, ξi) + dot(dual1[i, :], slack) <= epi[i])
        @constraint(m,  C'*dual1[i, :] .- B1 .<= lam .* W)
        @constraint(m, -C'*dual1[i, :] .+ B1 .<= lam .* W)

        @constraint(m, dot(B2, ξi) + dot(dual2[i, :], slack) <= epi[i])
        @constraint(m,  C'*dual2[i, :] .- B2 .<= lam .* W)
        @constraint(m, -C'*dual2[i, :] .+ B2 .<= lam .* W)
    end

    n_opts = Float64[]
    prof_is = Float64[]
    lam_opts = Float64[]

    @objective(m, Min, lam * eps + sum(epi) / N)
    optimize!(m)

    return value(n)
end

function run_experiments()
    Random.seed!(42)
    mkpath("results/extreme_wide_bounds")
    N_eps = length(CONFIG.epsilons)
    law_s = LocationScale(90.0, 35.0, TDist(3))
    law_d = MixtureModel([Normal(CONFIG.GMM.mu1, CONFIG.GMM.sig1), Normal(CONFIG.GMM.mu2, CONFIG.GMM.sig2)], [CONFIG.GMM.w1, CONFIG.GMM.w2])
    law_f, law_zeta = truncated(Normal(0.7, 0.2), 0.0, 1.0), Normal(0.0, 0.15)

    g_oos = clamp.((1.0 .+ rand(law_zeta, CONFIG.N_oos)) .* rand(law_f, CONFIG.N_oos), 0.0, 1.0)
    s_oos_b, d_oos_b = rand(law_s, CONFIG.N_oos), rand(law_d, CONFIG.N_oos)
    scenarios = [(name="pos", s_min=0.0), (name="neg", s_min=-50.0)]

    for N_train in TARGET_N_TRAINS
        println("\n>>> Running N_train = $N_train")
        results = Dict{Symbol, Any}[]

        for scen in scenarios
            s_oos = clamp.(s_oos_b, scen.s_min, CONFIG.Xi.s_max)
            d_oos = clamp.(d_oos_b, CONFIG.Xi.d_min, CONFIG.Xi.d_max)
            n_ro = solve_ro(scen.s_min)
            ro_metrics = compute_oos_metrics(n_ro, g_oos, s_oos, d_oos)

            saa_n, saa_metrics = zeros(CONFIG.N_sims), zeros(CONFIG.N_sims, 2)
            n_exact, insample_exact, metrics_exact, lam_exact = zeros(CONFIG.N_sims, N_eps), zeros(CONFIG.N_sims, N_eps), zeros(CONFIG.N_sims, N_eps, 2), zeros(CONFIG.N_sims, N_eps)
            n_mccormick, insample_mccormick, metrics_mccormick, lam_mccormick = zeros(CONFIG.N_sims, N_eps), zeros(CONFIG.N_sims, N_eps), zeros(CONFIG.N_sims, N_eps, 2), zeros(CONFIG.N_sims, N_eps)
            n_box, insample_box, metrics_box, lam_box = zeros(CONFIG.N_sims, N_eps), zeros(CONFIG.N_sims, N_eps), zeros(CONFIG.N_sims, N_eps, 2), zeros(CONFIG.N_sims, N_eps)

            for sim in 1:CONFIG.N_sims
                g_train = clamp.((1.0 .+ rand(law_zeta, N_train)) .* rand(law_f, N_train), 0.0, 1.0)
                s_train = clamp.(rand(law_s, N_train), scen.s_min, CONFIG.Xi.s_max)
                d_train = clamp.(rand(law_d, N_train), CONFIG.Xi.d_min, CONFIG.Xi.d_max)

                # exact
                W_ex = [CONFIG.W_mult.g / max(std(g_train), 1e-6),
                        CONFIG.W_mult.s / max(std(s_train), 1e-6),
                        CONFIG.W_mult.d / max(std(d_train), 1e-6)]

                # approx
                W_ap = [W_ex...,
                        1.0 / max(std(g_train .* s_train), 1e-6),
                        1.0 / max(std(g_train .* d_train), 1e-6)]

                saa_n[sim] = solve_saa(g_train, s_train, d_train)
                s_metrics = compute_oos_metrics(saa_n[sim], g_oos, s_oos, d_oos)
                saa_metrics[sim, :] .= [s_metrics.mean, s_metrics.cvar5]

                # n_exact[sim, :], insample_exact[sim, :], lam_exact[sim, :] = solve_dro_exact(g_train, s_train, d_train, scen.s_min, W_ex)
                # n_mccormick[sim, :], insample_mccormick[sim, :], lam_mccormick[sim, :] = solve_dro_approx_mccormick(g_train, s_train, d_train, scen.s_min, W_ap)
                n_box[sim, :], insample_box[sim, :], lam_box[sim, :] = solve_dro_approx_box(g_train, s_train, d_train, scen.s_min, W_ap)

                for j in 1:N_eps
                    # me = compute_oos_metrics(n_exact[sim, j], g_oos, s_oos, d_oos)
                    # ma = compute_oos_metrics(n_mccormick[sim, j], g_oos, s_oos, d_oos)
                    mbox = compute_oos_metrics(n_box[sim, j], g_oos, s_oos, d_oos)
                    # metrics_exact[sim, j, :] .= [me.mean, me.cvar5]
                    # metrics_mccormick[sim, j, :] .= [ma.mean, ma.cvar5]
                    metrics_box[sim, j, :] .= [mbox.mean, mbox.cvar5]
                end
            end

            # safety check
            println("Max violation McCormick > exact: ",
            maximum(insample_mccormick .- insample_exact))
            println("Max violation box > McCormick: ",
            maximum(insample_box .- insample_mccormick))

            for (j, eps) in enumerate(CONFIG.epsilons)
                push!(results, Dict(
                    :Scenario => scen.name,
                    :Epsilon => eps,
                    :Method => "exact",
                    :N_opt => mean(n_exact[:, j]),
                    :Profit_Mean => mean(metrics_exact[:, j, 1]),
                    :Reliability => mean(metrics_exact[:, j, 1] .>= insample_exact[:, j]),
                    :Profit_Q20 => quantile(metrics_exact[:, j, 1], 0.20),
                    :Profit_Q80 => quantile(metrics_exact[:, j, 1], 0.80),
                    :CVaR5 => mean(metrics_exact[:, j, 2]),
                    :SAA_Profit => mean(saa_metrics[:, 1]),
                    :RO_Profit => ro_metrics.mean,
                    :SAA_N => mean(saa_n),
                    :RO_N => n_ro,
                    :SAA_CVaR => mean(saa_metrics[:, 2]),
                    :RO_CVaR => ro_metrics.cvar5,
                    :Lambda => mean(lam_exact[:, j])
                ))
                push!(results, Dict(
                    :Scenario => scen.name,
                    :Epsilon => eps,
                    :Method => "approx-mccormick",
                    :N_opt => mean(n_mccormick[:, j]),
                    :Profit_Mean => mean(metrics_mccormick[:, j, 1]),
                    :Reliability => mean(metrics_mccormick[:, j, 1] .>= insample_mccormick[:, j]),
                    :Profit_Q20 => quantile(metrics_mccormick[:, j, 1], 0.20),
                    :Profit_Q80 => quantile(metrics_mccormick[:, j, 1], 0.80),
                    :CVaR5 => mean(metrics_mccormick[:, j, 2]),
                    :SAA_Profit => mean(saa_metrics[:, 1]),
                    :RO_Profit => ro_metrics.mean,
                    :SAA_N => mean(saa_n), :RO_N => n_ro,
                    :SAA_CVaR => mean(saa_metrics[:, 2]),
                    :RO_CVaR => ro_metrics.cvar5,
                    :Lambda => mean(lam_mccormick[:, j])
                ))
                push!(results, Dict(
                    :Scenario => scen.name,
                    :Epsilon => eps,
                    :Method => "approx-box",
                    :N_opt => mean(n_box[:, j]),
                    :Profit_Mean => mean(metrics_box[:, j, 1]),
                    :Reliability => mean(metrics_box[:, j, 1] .>= insample_box[:, j]),
                    :Profit_Q20 => quantile(metrics_box[:, j, 1], 0.20),
                    :Profit_Q80 => quantile(metrics_box[:, j, 1], 0.80),
                    :CVaR5 => mean(metrics_box[:, j, 2]),
                    :SAA_Profit => mean(saa_metrics[:, 1]),
                    :RO_Profit => ro_metrics.mean,
                    :SAA_N => mean(saa_n), :RO_N => n_ro,
                    :SAA_CVaR => mean(saa_metrics[:, 2]),
                    :RO_CVaR => ro_metrics.cvar5,
                    :Lambda => mean(lam_box[:, j])
                ))
            end
        end
        CSV.write("results/raw/dro_metrics_N_$(N_train).csv", DataFrame(results))
        println("Saved results/raw/dro_metrics_N_$(N_train).csv")
    end
end

function @main(ARGS)
    run_experiments()
end
