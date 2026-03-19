using JuMP
using HiGHS
using Distributions
using Plots
using LinearAlgebra
using Statistics
using LaTeXStrings

# style setup
default(
    fontfamily = "Computer Modern", 
    titlefontsize = 16,
    guidefontsize = 14,             
    tickfontsize = 12,             
    legendfontsize = 9,      
    linewidth = 2,                  
    framestyle = :box,             
    grid = true,
    gridalpha = 0.3,              
    margin = 5Plots.mm              
)

# out-of-sample procedure
function compute_oos_profit(n::Float64, g::Vector{Float64}, S::Vector{Float64}, PREP::Vector{Float64}, PREN::Vector{Float64})
    profits = S .* n .+ PREP .* max.(g .- n, 0.0) .- PREN .* max.(n .- g, 0.0)
    return mean(profits)
end


function solve_empirical_saa(g_train, S_train, PREP_train, PREN_train)
    N = length(g_train)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    @variables(model, begin
        0 <= n <= 1
        s_loss[1:N]  
    end)
    
    for i in 1:N
        g_i, S_i, PREP_i, PREN_i = g_train[i], S_train[i], PREP_train[i], PREN_train[i]
        @constraint(model, s_loss[i] >= -S_i * n + PREP_i * (n - g_i)) 
        @constraint(model, s_loss[i] >= -S_i * n - PREN_i * (g_i - n)) 
    end
    
    @objective(model, Min, sum(s_loss) / N)
    optimize!(model)
    
    return value(n), objective_value(model)
end


function solve_dro(g_train, S_train, PREP_train, PREN_train, 
                   epsilons, S_min, S_max, PREP_min, PREP_max, PREN_min, PREN_max)
    N = length(g_train)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    # support bounds: C * [g; S; PREP; PREN] <= d
    C = [ 1.0  0.0  0.0  0.0; 
         -1.0  0.0  0.0  0.0;
          0.0  1.0  0.0  0.0; 
          0.0 -1.0  0.0  0.0;
          0.0  0.0  1.0  0.0;
          0.0  0.0 -1.0  0.0;
          0.0  0.0  0.0  1.0;
          0.0  0.0  0.0 -1.0]
    d = [1.0, 0.0, S_max, -S_min, PREP_max, -PREP_min, PREN_max, -PREN_min]
    
    # scaling 
    W = [1.0, 1.0/(S_max - S_min), 1.0/(PREP_max - PREP_min), 1.0/(PREN_max - PREN_min)] 
    
    @variables(model, begin
        0 <= n <= 1            
        lambda >= 0            # dual variable for Wasserstein radius 
        s_epi[1:N]             # epigraphic variables
        gamma1[1:N, 1:8] >= 0  # dual variable for surplus support constraints
        gamma2[1:N, 1:8] >= 0  # dual variable for deficit support constraints
    end)
    
    for i in 1:N
        g_i, S_i, PREP_i, PREN_i = g_train[i], S_train[i], PREP_train[i], PREN_train[i]
        slack = d .- C * [g_i, S_i, PREP_i, PREN_i]
        
        # gradients of the loss w.r.t the uncertainties
        B1 = [-PREP_i, -n, n - g_i, 0.0] 
        B2 = [-PREN_i, -n, 0.0, n - g_i] 
        
        # surplus constraints (g > n)
        @constraint(model, (-S_i * n + PREP_i * (n - g_i)) + dot(gamma1[i, :], slack) <= s_epi[i])
        @constraint(model,  C' * gamma1[i, :] .- B1 .<= lambda .* W)
        @constraint(model, -(C' * gamma1[i, :] .- B1) .<= lambda .* W)
        
        # deficit constraints (g < n)
        @constraint(model, (-S_i * n - PREN_i * (g_i - n)) + dot(gamma2[i, :], slack) <= s_epi[i])
        @constraint(model,  C' * gamma2[i, :] .- B2 .<= lambda .* W)
        @constraint(model, -(C' * gamma2[i, :] .- B2) .<= lambda .* W)
    end
    
    n_opts = zeros(length(epsilons))
    costs  = zeros(length(epsilons))
    
    # update the objective
    for (idx, eps) in enumerate(epsilons)
        @objective(model, Min, lambda * eps + sum(s_epi) / N)
        optimize!(model)
        n_opts[idx] = value(n)
        costs[idx] = objective_value(model)
    end
    
    return n_opts, costs
end

# simulation params
N_sims = 200        
N_train = 30        
N_oos = 100_000     
epsilons = 10 .^ range(-4, stop=-0.5, length=100)

# params
mu_g, sigma_g = 0.7, 0.25
mu_S, tau_S = 65.0, 30.0
S_max = 400.0
S_min_pos, S_min_neg = 0.0, -100.0
PREP_min, PREP_max = -1000.0, 1000.0
PREN_min, PREN_max = -1000.0, 1000.0

# g and S sampled independently
law_g = Normal(mu_g, sigma_g)
law_S = LocationScale(mu_S, tau_S, TDist(3))
penalties = MixtureModel([Normal(5.0, 2.0), Normal(600.0, 50.0)], [0.95, 0.05])

# generate out-of-sample data
g_oos = clamp.(rand(law_g, N_oos), 0.0, 1.0)
S_oos_base = rand(law_S, N_oos)
surp_oos, def_oos = rand(penalties, N_oos), rand(penalties, N_oos)
S_oos_pos = clamp.(S_oos_base, S_min_pos, S_max)
PREP_oos_pos = clamp.(S_oos_pos .- surp_oos, PREP_min, PREP_max)
PREN_oos_pos = clamp.(S_oos_pos .+ def_oos, PREN_min, PREN_max)
S_oos_neg = clamp.(S_oos_base, S_min_neg, S_max)
PREP_oos_neg = clamp.(S_oos_neg .- surp_oos, PREP_min, PREP_max)
PREN_oos_neg = clamp.(S_oos_neg .+ def_oos, PREN_min, PREN_max)

profit_is_pos, profit_oos_pos, n_opt_pos = (zeros(length(epsilons), N_sims) for _ in 1:3)
profit_is_neg, profit_oos_neg, n_opt_neg = (zeros(length(epsilons), N_sims) for _ in 1:3)

saa_profit_oos_pos, saa_n_opt_pos = zeros(N_sims), zeros(N_sims)
saa_profit_oos_neg, saa_n_opt_neg = zeros(N_sims), zeros(N_sims)

println("Starting $N_sims simulations with N_train=$N_train")
for s in 1:N_sims
    if s % 10 == 0; println("$s / $N_sims done"); end
    
    g_tr = clamp.(rand(law_g, N_train), 0.0, 1.0)
    S_tr_base = rand(law_S, N_train)
    surp_tr, def_tr = rand(penalties, N_train), rand(penalties, N_train)
    
    # 1°) positive prices only
    S_tr_pos = clamp.(S_tr_base, S_min_pos, S_max)
    PREP_tr_pos = clamp.(S_tr_pos .- surp_tr, PREP_min, PREP_max)
    PREN_tr_pos = clamp.(S_tr_pos .+ def_tr, PREN_min, PREN_max)
    
    # SAA
    n_saa_pos, _ = solve_empirical_saa(g_tr, S_tr_pos, PREP_tr_pos, PREN_tr_pos)
    saa_n_opt_pos[s] = n_saa_pos
    saa_profit_oos_pos[s] = compute_oos_profit(n_saa_pos, g_oos, S_oos_pos, PREP_oos_pos, PREN_oos_pos)

    # DRO
    n_pos, cost_pos = solve_dro(g_tr, S_tr_pos, PREP_tr_pos, PREN_tr_pos, epsilons, S_min_pos, S_max, PREP_min, PREP_max, PREN_min, PREN_max)
    profit_is_pos[:, s] = -cost_pos
    n_opt_pos[:, s] = n_pos
    profit_oos_pos[:, s] = [compute_oos_profit(n, g_oos, S_oos_pos, PREP_oos_pos, PREN_oos_pos) for n in n_pos]

    # 2°) negative prices allowed
    S_tr_neg = clamp.(S_tr_base, S_min_neg, S_max)
    PREP_tr_neg = clamp.(S_tr_neg .- surp_tr, PREP_min, PREP_max)
    PREN_tr_neg = clamp.(S_tr_neg .+ def_tr, PREN_min, PREN_max)
    
    # SAA
    n_saa_neg, _ = solve_empirical_saa(g_tr, S_tr_neg, PREP_tr_neg, PREN_tr_neg)
    saa_n_opt_neg[s] = n_saa_neg
    saa_profit_oos_neg[s] = compute_oos_profit(n_saa_neg, g_oos, S_oos_neg, PREP_oos_neg, PREN_oos_neg)

    # DRO
    n_neg, cost_neg = solve_dro(g_tr, S_tr_neg, PREP_tr_neg, PREN_tr_neg, epsilons, S_min_neg, S_max, PREP_min, PREP_max, PREN_min, PREN_max)
    profit_is_neg[:, s] = -cost_neg
    n_opt_neg[:, s] = n_neg
    profit_oos_neg[:, s] = [compute_oos_profit(n, g_oos, S_oos_neg, PREP_oos_neg, PREN_oos_neg) for n in n_neg]
end

mean_p_oos_pos = mean(profit_oos_pos, dims=2)[:, 1]
q20_p_oos_pos = [quantile(profit_oos_pos[i, :], 0.20) for i in 1:length(epsilons)]
q80_p_oos_pos = [quantile(profit_oos_pos[i, :], 0.80) for i in 1:length(epsilons)]
mean_n_pos = mean(n_opt_pos, dims=2)[:, 1]

mean_p_oos_neg = mean(profit_oos_neg, dims=2)[:, 1]
q20_p_oos_neg = [quantile(profit_oos_neg[i, :], 0.20) for i in 1:length(epsilons)]
q80_p_oos_neg = [quantile(profit_oos_neg[i, :], 0.80) for i in 1:length(epsilons)]
mean_n_neg = mean(n_opt_neg, dims=2)[:, 1]

# Reliability metric: proba that the out-of-sample profit exceeds the 
# worst-case in-sample guarantee 
rel_pos = [mean(profit_oos_pos[i, :] .>= profit_is_pos[i, :]) for i in 1:length(epsilons)]
rel_neg = [mean(profit_oos_neg[i, :] .>= profit_is_neg[i, :]) for i in 1:length(epsilons)]

# SAA baseline
empirical_saa_profit_pos = mean(saa_profit_oos_pos)
empirical_saa_profit_neg = mean(saa_profit_oos_neg)
empirical_saa_n_pos = mean(saa_n_opt_pos)
empirical_saa_n_neg = mean(saa_n_opt_neg)

# plots
my_xticks = 10.0 .^ (-4:1:2)

p1 = plot(epsilons, mean_p_oos_pos, 
          ribbon=(mean_p_oos_pos .- q20_p_oos_pos, q80_p_oos_pos .- mean_p_oos_pos),
          fillalpha=0.2, color=:dodgerblue, lw=2, marker=:dtriangle, label="DRO-pos (out-of-sample)",
          xscale=:log10, xticks=my_xticks, xminorgrid=true, minorgridalpha=0.1,
          xlabel=L"Radius $\epsilon$", ylabel="Profit",
          legend=:bottomleft)

plot!(p1, epsilons, mean_p_oos_neg, 
      ribbon=(mean_p_oos_neg .- q20_p_oos_neg, q80_p_oos_neg .- mean_p_oos_neg),
      fillalpha=0.2, color=:indianred1, lw=2, marker=:dtriangle, label="DRO-neg (out-of-sample)")

hline!(p1, [empirical_saa_profit_pos], label="Empirical SAA-pos", color=:dodgerblue, ls=:dash, lw=2)
hline!(p1, [empirical_saa_profit_neg], label="Empirical SAA-neg", color=:indianred1, ls=:dash, lw=2)

p1_twin = twinx(p1)
plot!(p1_twin, epsilons, rel_pos, 
      lw=2, color=:firebrick, ls=:dashdot, label="Reliability", 
      ylabel="Reliability", ylims=(0.0, 1.05),
      legend=:bottomright)

p2 = plot(epsilons, mean_n_pos, lw=2, marker=:+, color=:dodgerblue, label="DRO-pos",
          xscale=:log10, xticks=my_xticks, xminorgrid=true, minorgridalpha=0.1,
          xlabel=L"Radius $\epsilon$", ylabel=L"Optimal nomination $n^{\star}$", 
          ylims=(0.0, 1.0), legend=:bottomleft)

plot!(p2, epsilons, mean_n_neg, lw=2, marker=:+, color=:indianred1, label="DRO-neg")

hline!(p2, [empirical_saa_n_pos], label="Empirical SAA-pos", color=:dodgerblue, ls=:dash, lw=2)
hline!(p2, [empirical_saa_n_neg], label="Empirical SAA-neg", color=:indianred1, ls=:dash, lw=2)

display(p1)
display(p2)

savefig(p1, "oos_profit_sample_stochastic_imbal_$N_train.pdf")
savefig(p2, "oos_decision_stochastic_imbal_$N_train.pdf")
