using JuMP
using HiGHS
using Distributions
using Plots
using LinearAlgebra
using Statistics
using LaTeXStrings
using Random

# style setup
default(
    fontfamily = "Computer Modern", 
    titlefontsize = 16, guidefontsize = 14,             
    tickfontsize = 12, legendfontsize = 9,      
    linewidth = 2, framestyle = :box,             
    grid = true, gridalpha = 0.3,              
    margin = 5Plots.mm              
)

# out-of-sample procedure
function compute_oos_profit(n::Float64, g::Vector{Float64}, s::Float64, delta::Float64, tau::Float64)
    c_plus = (s + delta) * tau - delta
    c_minus = (s + delta) * tau + delta
    profits = (g .* s) .- c_plus .* max.(g .- n, 0.0) .- c_minus .* max.(n .- g, 0.0)
    return mean(profits)
end

# polyhedral support set Xi = {C * [g] <= d}
function build_support_set()
    C = reshape([1.0, -1.0], 2, 1)  # g <= 1, -g <= 0
    d = [1.0, 0.0]
    return C, d
end

function solve_empirical_saa(g_train, s, delta, tau)
    N = length(g_train)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    @variables(model, begin 
        0 <= n <= 1 
        s_loss[1:N] 
    end)
    
    c_plus = (s + delta) * tau - delta
    c_minus = (s + delta) * tau + delta

    for i in 1:N
        @constraint(model, s_loss[i] >= -g_train[i]*s + c_plus * (g_train[i] - n)) 
        @constraint(model, s_loss[i] >= -g_train[i]*s + c_minus * (n - g_train[i])) 
    end
    
    @objective(model, Min, sum(s_loss) / N)
    optimize!(model)
    
    return value(n), objective_value(model)
end

function solve_ro_closed(s, delta, tau)
    c_plus = (s + delta) * tau - delta
    c_minus = (s + delta) * tau + delta
    
    n_raw = (c_plus - s) / (c_plus + c_minus)
    n_opt = clamp(n_raw, 0.0, 1.0)
    
    wc_loss = max(c_minus * n_opt, c_plus - s - c_plus * n_opt)
    
    return n_opt, -wc_loss
end

function solve_ro(s, delta, tau)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    C, d = build_support_set()
    
    @variables(model, begin 
        0 <= n <= 1 
        wc_loss
        gamma1[1:2] >= 0
        gamma2[1:2] >= 0 
    end)
    
    c_plus = (s + delta) * tau - delta
    c_minus = (s + delta) * tau + delta

    # Gradients w.r.t [g]
    B1 = [-s + c_plus]
    B2 = [-s - c_minus]
    
    @constraint(model, C' * gamma1 .== B1)
    @constraint(model, dot(d, gamma1) - c_plus * n <= wc_loss)
    
    @constraint(model, C' * gamma2 .== B2)
    @constraint(model, dot(d, gamma2) + c_minus * n <= wc_loss)
    
    @objective(model, Min, wc_loss)
    optimize!(model)
    
    return value(n), -objective_value(model) 
end

function solve_dro(g_train, s, delta, epsilons, W, tau_param)
    N = length(g_train)
    model = Model(HiGHS.Optimizer); set_silent(model)

    C, d = build_support_set()
    
    @variables(model, begin 
        0 <= n <= 1
        lambda >= 0
        s_epi[1:N]
        gamma1[1:N, 1:2] >= 0
        gamma2[1:N, 1:2] >= 0 
    end)
    
    c_plus = (s + delta) * tau_param - delta
    c_minus = (s + delta) * tau_param + delta

    B1 = [-s + c_plus]
    B2 = [-s - c_minus]

    for i in 1:N
        xi_i = [g_train[i]] 
        slack = d .- C * xi_i
        
        @constraint(model, -g_train[i]*s + c_plus*(g_train[i] - n) + dot(gamma1[i, :], slack) <= s_epi[i])
        @constraint(model,  C' * gamma1[i, :] .- B1 .<= lambda * W)
        @constraint(model, -(C' * gamma1[i, :] .- B1) .<= lambda * W)
        
        @constraint(model, -g_train[i]*s + c_minus*(n - g_train[i]) + dot(gamma2[i, :], slack) <= s_epi[i])
        @constraint(model,  C' * gamma2[i, :] .- B2 .<= lambda * W)
        @constraint(model, -(C' * gamma2[i, :] .- B2) .<= lambda * W)
    end
    
    n_opts, costs = zeros(length(epsilons)), zeros(length(epsilons))
    for (idx, eps) in enumerate(epsilons)
        @objective(model, Min, lambda * eps + sum(s_epi) / N)
        optimize!(model)
        n_opts[idx], costs[idx] = value(n), objective_value(model)
    end
    return n_opts, costs
end

# simulation params
Random.seed!(42)
k = 1.2
tau_val = (k - 1.0) / (k + 1.0)
N_sims = 100
N_train = 30
N_oos = 100_000     
epsilons = 10 .^ range(-4, stop=1, length=100) 

law_f = truncated(Normal(0.7, 0.2), 0.0, 1.0)
law_zeta = Normal(0.0, 0.15)

# deterministic prices
s_val = 65.0 
delta_val = 5.0 

# Generate test data
f_oos = rand(law_f, N_oos)
zeta_oos = rand(law_zeta, N_oos)
g_oos = clamp.((1.0 .+ zeta_oos) .* f_oos, 0.0, 1.0)

c_plus = (s_val + delta_val) * tau_val - delta_val
c_minus = (s_val + delta_val) * tau_val + delta_val
r_minus = s_val - c_plus
r_plus  = s_val + c_minus

# Theoretical closed-form solution
target_prob = (s_val - r_minus) / (r_plus - r_minus)
n_theory = quantile(g_oos, target_prob) 
profit_theory = compute_oos_profit(n_theory, g_oos, s_val, delta_val, tau_val)

profit_is = zeros(length(epsilons), N_sims)
profit_oos = zeros(length(epsilons), N_sims)
n_opt = zeros(length(epsilons), N_sims)
saa_profit_oos = zeros(N_sims)
saa_n_opt = zeros(N_sims)

# RO baseline
ro_n, _ = solve_ro_closed(s_val, delta_val, tau_val)
ro_profit_oos = compute_oos_profit(ro_n, g_oos, s_val, delta_val, tau_val)

println("Starting $N_sims simulations with N_train=$N_train")
for sim in 1:N_sims
    if sim % 10 == 0; println("$sim / $N_sims done"); end

    # Generate training data
    f_tr = rand(law_f, N_train)
    zeta_tr = rand(law_zeta, N_train)
    g_tr = clamp.((1.0 .+ zeta_tr) .* f_tr, 0.0, 1.0)

    # scaling
    W = 1.0 / max(std(g_tr), 1e-6)  

    # Empirical SAA
    saa_n_opt[sim], _ = solve_empirical_saa(g_tr, s_val, delta_val, tau_val)
    saa_profit_oos[sim] = compute_oos_profit(saa_n_opt[sim], g_oos, s_val, delta_val, tau_val)

    # DRO
    n_dro, cost_dro = solve_dro(g_tr, s_val, delta_val, epsilons, W, tau_val)
    profit_is[:, sim] = -cost_dro
    n_opt[:, sim] = n_dro
    profit_oos[:, sim] = [compute_oos_profit(n, g_oos, s_val, delta_val, tau_val) for n in n_dro]
end

# aggregate statistics across all MC runs
mean_p_oos = mean(profit_oos, dims=2)[:, 1]
q20_p_oos = [quantile(profit_oos[j, :], 0.20) for j in 1:length(epsilons)]
q80_p_oos = [quantile(profit_oos[j, :], 0.80) for j in 1:length(epsilons)]
mean_n = mean(n_opt, dims=2)[:, 1]

# Reliability metric: proba that out-of-sample profit exceeds in-sample worst-case guarantee
rel = [mean(profit_oos[j, :] .>= profit_is[j, :]) for j in 1:length(epsilons)]

my_xticks = 10.0 .^ (-4:1:1)

# profit plot
p1 = plot(epsilons, mean_p_oos, ribbon=(mean_p_oos .- q20_p_oos, q80_p_oos .- mean_p_oos),
          fillalpha=0.15, color=:dodgerblue, lw=2, marker=:dtriangle, label="DRO",
          xscale=:log10, xticks=my_xticks, minorgrid=true, minorgridalpha=0.1,
          xlabel=L"Radius $\epsilon$", ylabel="Profit [EUR/MWh]", legend=:bottomleft)
hline!(p1, [mean(saa_profit_oos)], label="SAA", color=:dodgerblue, ls=:dash, lw=2)
hline!(p1, [ro_profit_oos], label="RO", color=:dodgerblue, ls=:dot, lw=2)
hline!(p1, [profit_theory], label="Theoretical", color=:forestgreen, ls=:solid, lw=2)

plot!(twinx(p1), epsilons, rel, lw=2, color=:firebrick, ls=:dashdot, label="Reliability", 
      xscale=:log10, xticks=my_xticks, grid=false, ylabel="Reliability", ylims=(0.0, 1.05), legend=:bottomright)

# decision plot
p2 = plot(epsilons, mean_n, lw=2, marker=:dtriangle, color=:dodgerblue, label="DRO",
          xscale=:log10, xticks=my_xticks, minorgrid=true, minorgridalpha=0.1,
          xlabel=L"Radius $\epsilon$", ylabel=L"Optimal nomination $n^{\star}$", 
          ylims=(0.0, 1.0), legend=:topleft)
hline!(p2, [mean(saa_n_opt)], label="SAA", color=:dodgerblue, ls=:dash, lw=2)
hline!(p2, [ro_n], label="RO", color=:dodgerblue, ls=:dot, lw=2)
hline!(p2, [n_theory], label="Theoretical", color=:forestgreen, ls=:solid, lw=2)

display(p1)
display(p2)

# save plots
output_dir = "plots/newsvendor_baseline"
mkpath(output_dir)
savefig(p1, joinpath(output_dir, "oos_profit_newsvendor_$(N_train).pdf"))
savefig(p2, joinpath(output_dir, "oos_decision_newsvendor_$(N_train).pdf"))
