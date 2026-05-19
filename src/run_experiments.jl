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
    Xi = (s_max = 500.0, d_min = -1000.0, d_max = 1000.0),
    GMM = (mu1 = -20.0, sig1 = 50.0, w1 = 0.95, mu2 = 1000.0, sig2 = 1.0, w2 = 0.05),
    W_mult = (g = 1.0, s = 1.0, d = 1.0) 
)
const tau_val = (CONFIG.k_val - 1.0) / (CONFIG.k_val + 1.0)

function compute_oos_metrics(n, g, s, delta)
    cp, cm = (s .+ delta) .* tau_val .- delta, (s .+ delta) .* tau_val .+ delta
    profits = sort!((g .* s) .- cp .* max.(g .- n, 0.0) .- cm .* max.(n .- g, 0.0))
    idx_5 = max(1, round(Int, 0.05 * length(profits)))
    return (mean = mean(profits), cvar5 = mean(view(profits, 1:idx_5)))
end

function build_support_set(s_min)
    s_max, d_min, d_max = CONFIG.Xi.s_max, CONFIG.Xi.d_min, CONFIG.Xi.d_max
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

function solve_saa(g_tr, s_tr, d_tr)
    m = Model(HiGHS.Optimizer); set_silent(m)
    N = length(g_tr)
    @variables(m, begin 
        0 <= n <= 1
        loss[1:N] 
    end)
    for i in 1:N
        cp, cm = (s_tr[i]+d_tr[i])*tau_val - d_tr[i], (s_tr[i]+d_tr[i])*tau_val + d_tr[i]
        @constraint(m, loss[i] >= -g_tr[i]*s_tr[i] + cp*(g_tr[i] - n))
        @constraint(m, loss[i] >= -g_tr[i]*s_tr[i] + cm*(n - g_tr[i]))
    end
    @objective(m, Min, sum(loss)/N); optimize!(m); return value(n)
end

function solve_ro(s_min)
    m = Model(HiGHS.Optimizer); set_silent(m)
    C, d = build_support_set(s_min)
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

function solve_dro_exact(g_tr, s_tr, d_tr, s_min, W)
    N = length(g_tr)
    m = Model(() -> Gurobi.Optimizer(GRB_ENV)); set_silent(m)
    @variables(m, begin 
        0 <= n <= 1
        lam >= 0
        epi[1:N] 
    end)
    for i in 1:N
        for g_k in [0.0, g_tr[i], 1.0], s_k in [s_min, s_tr[i], CONFIG.Xi.s_max], d_k in [CONFIG.Xi.d_min, d_tr[i], CONFIG.Xi.d_max]
            dist = W[1]*abs(g_k-g_tr[i]) + W[2]*abs(s_k-s_tr[i]) + W[3]*abs(d_k-d_tr[i])
            cp, cm = (s_k+d_k)*tau_val - d_k, (s_k+d_k)*tau_val + d_k
            @constraint(m, epi[i] >= -g_k*s_k + cp*(g_k - n) - lam*dist)
            @constraint(m, epi[i] >= -g_k*s_k + cm*(n - g_k) - lam*dist)
        end
    end

    n_opts, prof_is = Float64[], Float64[]

    for eps in CONFIG.epsilons
        @objective(m, Min, lam*eps + sum(epi)/N)
        optimize!(m)
        push!(n_opts, value(n))
        push!(prof_is, -objective_value(m))
    end
    return n_opts, prof_is
end

function solve_dro_approx(g_tr, s_tr, d_tr, s_min, W)
    N = length(g_tr)
    m = Model(HiGHS.Optimizer); set_silent(m)
    C, d = build_support_set(s_min)
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
        slack = d .- C * [g_tr[i], s_tr[i], d_tr[i], g_tr[i]*s_tr[i], g_tr[i]*d_tr[i]]
        cp, cm = (s_tr[i]+d_tr[i])*tau_val - d_tr[i], (s_tr[i]+d_tr[i])*tau_val + d_tr[i]
        @constraint(m, -g_tr[i]*s_tr[i] + cp*(g_tr[i]-n) + dot(g1[i,:], slack) <= epi[i])
        @constraint(m, C'*g1[i,:] .- B1 .<= lam .* W)
        @constraint(m, -(C'*g1[i,:] .- B1) .<= lam .* W)
        @constraint(m, -g_tr[i]*s_tr[i] + cm*(n-g_tr[i]) + dot(g2[i,:], slack) <= epi[i])
        @constraint(m, C'*g2[i,:] .- B2 .<= lam .* W)
        @constraint(m, -(C'*g2[i,:] .- B2) .<= lam .* W)
    end
    n_opts, prof_is = Float64[], Float64[]
    for eps in CONFIG.epsilons
        @objective(m, Min, lam*eps + sum(epi)/N)
        optimize!(m)
        push!(n_opts, value(n))
        push!(prof_is, -objective_value(m))
    end
    return n_opts, prof_is
end

function run_experiments()
    Random.seed!(42)
    mkpath("results/raw")
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
            ro_met = compute_oos_metrics(n_ro, g_oos, s_oos, d_oos)

            saa_n, saa_mets = zeros(CONFIG.N_sims), zeros(CONFIG.N_sims, 2)
            ex_n, ex_is, ex_mets = zeros(CONFIG.N_sims, 100), zeros(CONFIG.N_sims, 100), zeros(CONFIG.N_sims, 100, 2)
            ap_n, ap_is, ap_mets = zeros(CONFIG.N_sims, 100), zeros(CONFIG.N_sims, 100), zeros(CONFIG.N_sims, 100, 2)

            for sim in 1:CONFIG.N_sims
                g_tr = clamp.((1.0 .+ rand(law_zeta, N_train)) .* rand(law_f, N_train), 0.0, 1.0)
                s_tr = clamp.(rand(law_s, N_train), scen.s_min, CONFIG.Xi.s_max)
                d_tr = clamp.(rand(law_d, N_train), CONFIG.Xi.d_min, CONFIG.Xi.d_max)

                # exact
                W_ex = [CONFIG.W_mult.g / max(std(g_tr), 1e-6), 
                        CONFIG.W_mult.s / max(std(s_tr), 1e-6), 
                        CONFIG.W_mult.d / max(std(d_tr), 1e-6)]
                
                # approx
                W_ap = [W_ex..., 
                        1.0 / max(std(g_tr .* s_tr), 1e-6), 
                        1.0 / max(std(g_tr .* d_tr), 1e-6)]

                saa_n[sim] = solve_saa(g_tr, s_tr, d_tr)
                s_met = compute_oos_metrics(saa_n[sim], g_oos, s_oos, d_oos)
                saa_mets[sim, :] .= [s_met.mean, s_met.cvar5]

                ex_n[sim, :], ex_is[sim, :] = solve_dro_exact(g_tr, s_tr, d_tr, scen.s_min, W_ex)
                ap_n[sim, :], ap_is[sim, :] = solve_dro_approx(g_tr, s_tr, d_tr, scen.s_min, W_ap)

                for j in 1:100
                    me = compute_oos_metrics(ex_n[sim, j], g_oos, s_oos, d_oos)
                    ma = compute_oos_metrics(ap_n[sim, j], g_oos, s_oos, d_oos)
                    ex_mets[sim, j, :] .= [me.mean, me.cvar5]
                    ap_mets[sim, j, :] .= [ma.mean, ma.cvar5]
                end
            end

            for (j, eps) in enumerate(CONFIG.epsilons)
                push!(results, Dict(
                    :Scenario => scen.name, :Epsilon => eps, :Method => "exact", :N_opt => mean(ex_n[:, j]), 
                    :Profit_Mean => mean(ex_mets[:, j, 1]), :Reliability => mean(ex_mets[:, j, 1] .>= ex_is[:, j]),
                    :Profit_Q20 => quantile(ex_mets[:, j, 1], 0.20), :Profit_Q80 => quantile(ex_mets[:, j, 1], 0.80),
                    :CVaR5 => mean(ex_mets[:, j, 2]), :SAA_Profit => mean(saa_mets[:, 1]), :RO_Profit => ro_met.mean,
                    :SAA_N => mean(saa_n), :RO_N => n_ro, :SAA_CVaR => mean(saa_mets[:, 2]), :RO_CVaR => ro_met.cvar5
                ))
                push!(results, Dict(
                    :Scenario => scen.name, :Epsilon => eps, :Method => "approx", :N_opt => mean(ap_n[:, j]), 
                    :Profit_Mean => mean(ap_mets[:, j, 1]), :Reliability => mean(ap_mets[:, j, 1] .>= ap_is[:, j]),
                    :Profit_Q20 => quantile(ap_mets[:, j, 1], 0.20), :Profit_Q80 => quantile(ap_mets[:, j, 1], 0.80),
                    :CVaR5 => mean(ap_mets[:, j, 2]), :SAA_Profit => mean(saa_mets[:, 1]), :RO_Profit => ro_met.mean,
                    :SAA_N => mean(saa_n), :RO_N => n_ro, :SAA_CVaR => mean(saa_mets[:, 2]), :RO_CVaR => ro_met.cvar5
                ))
            end
        end
        CSV.write("results/raw/dro_metrics_N_$(N_train).csv", DataFrame(results))
        println("Saved results/raw/dro_metrics_N_$(N_train).csv")
    end
end

run_experiments()
