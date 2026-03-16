using JuMP
using HiGHS
using DataFrames
using Distributions
using Plots
using LinearAlgebra
using LaTeXStrings

# style setup 
default(
    fontfamily = "Computer Modern", 
    titlefontsize = 18,
    guidefontsize = 18,             
    tickfontsize = 16,             
    legendfontsize = 12,          
    linewidth = 2,                  
    framestyle = :box,             
    grid = true,
    gridalpha = 0.3,              
    margin = 5Plots.mm              
)

function solve_dro_4D(g_data::Vector{Float64}, S_data::Vector{Float64}, 
    PREP_data::Vector{Float64}, PREN_data::Vector{Float64},
    epsilon::Float64, 
    S_min::Float64, S_max::Float64,
    PREP_min::Float64, PREP_max::Float64,
    PREN_min::Float64, PREN_max::Float64)
    N = length(g_data)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    # support bounds: C * [g; S; PREP; PREN] <= d
    C = [1.0 0.0 0.0 0.0; 
        -1.0 0.0 0.0 0.0;
         0.0 1.0 0.0 0.0; 
         0.0 -1.0 0.0 0.0;
         0.0 0.0 1.0 0.0;
         0.0 0.0 -1.0 0.0;
         0.0 0.0 0.0 1.0;
         0.0 0.0 0.0 -1.0;
        ]
        d = [1.0, 0.0, S_max, -S_min, PREP_max, -PREP_min, PREN_max, -PREN_min]
    
    # scaling
    W_weights = [1.0, 1.0 / (S_max - S_min), 1.0 / (PREP_max - PREP_min), 1.0 / (PREN_max - PREN_min)] 
    
    @variables(model, begin
        0 <= n <= 1            
        lambda >= 0            
        s_epi[1:N]             
        gamma1[1:N, 1:8] >= 0  
        gamma2[1:N, 1:8] >= 0  
    end)
    
    for i in 1:N
        g_i, S_i, PREP_i, PREN_i = g_data[i], S_data[i], PREP_data[i], PREN_data[i]
        slack = d .- C * [g_i, S_i, PREP_i, PREN_i]
        
        # surplus (g > n)
        loss1_val = -S_i * n + PREP_i * (n - g_i)
        B1 = [-PREP_i, -n, n - g_i, 0.0] 
        
        @constraint(model, loss1_val + dot(gamma1[i, :], slack) <= s_epi[i])
        @constraint(model, C' * gamma1[i, :] .- B1 .<= lambda .* W_weights)
        @constraint(model, -(C' * gamma1[i, :] .- B1) .<= lambda .* W_weights)
        
        # deficit (g < n)
        loss2_val = -S_i * n - PREN_i * (g_i - n)
        B2 = [-PREN_i, -n, 0.0, n - g_i] 
        
        @constraint(model, loss2_val + dot(gamma2[i, :], slack) <= s_epi[i])
        @constraint(model, C' * gamma2[i, :] .- B2 .<= lambda .* W_weights)
        @constraint(model, -(C' * gamma2[i, :] .- B2) .<= lambda .* W_weights)
    end
    
    @objective(model, Min, lambda * epsilon + sum(s_epi) / N)
    optimize!(model)
    
    return value(n), objective_value(model)
end

# params
N_samples = 1000
mu_g, sigma_g = 0.7, 0.1
mu_S, tau_S = 65.0, 25.0
S_max = 400.0
PREP_min, PREP_max = -100.0, 300.0
PREN_min, PREN_max = -50.0, 400.0
epsilons = 0.0:0.02:2.0 

law_g = Normal(mu_g, sigma_g)
law_S = LocationScale(mu_S, tau_S, TDist(3))

# base unbounded samples 
g_base = clamp.(rand(law_g, N_samples), 0.0, 1.0)
S_base = rand(law_S, N_samples)

penalty_regimes = MixtureModel([Normal(5.0, 2.0), Normal(100.0, 50.0)], [0.8, 0.2])
surplus_penalty_samples = rand(penalty_regimes, N_samples)
deficit_penalty_samples = rand(penalty_regimes, N_samples)

# no negative prices allowed
S_min_pos = 0.0
g_samples_pos = copy(g_base)
S_samples_pos = clamp.(S_base, S_min_pos, S_max)

PREP_samples_pos = clamp.(S_samples_pos .- surplus_penalty_samples, PREP_min, PREP_max)
PREN_samples_pos = clamp.(S_samples_pos .+ deficit_penalty_samples, PREN_min, PREN_max)

mean_S_pos = mean(S_samples_pos)
mean_PREP_pos = mean(PREP_samples_pos)
mean_PREN_pos = mean(PREN_samples_pos)

target_prob_pos = (mean_S_pos - mean_PREP_pos) / (mean_PREN_pos - mean_PREP_pos)
n_theory_pos = quantile(law_g, target_prob_pos)

profit_theory_pos = mean(
    S_samples_pos .* n_theory_pos .- max.(
        PREP_samples_pos .* (n_theory_pos .- g_samples_pos), 
        PREN_samples_pos .* (n_theory_pos .- g_samples_pos)
    )
)

results_pos = DataFrame(eps = Float64[], n_opt = Float64[], profit = Float64[])
for eps in epsilons
    n_opt, cost = solve_dro_4D(g_samples_pos, S_samples_pos, PREP_samples_pos, PREN_samples_pos, eps, S_min_pos, S_max, PREP_min, PREP_max, PREN_min, PREN_max)    
    push!(results_pos, (eps = eps, n_opt = n_opt, profit = -cost))
end

# negative prices allowed
S_min_neg = -50.0
g_samples_neg = copy(g_base)
S_samples_neg = clamp.(S_base, S_min_neg, S_max)

PREP_samples_neg = clamp.(S_samples_neg .- surplus_penalty_samples, PREP_min, PREP_max)
PREN_samples_neg = clamp.(S_samples_neg .+ deficit_penalty_samples, PREN_min, PREN_max)

mean_S_neg = mean(S_samples_neg)
mean_PREP_neg = mean(PREP_samples_neg)
mean_PREN_neg = mean(PREN_samples_neg)

target_prob_neg = (mean_S_neg - mean_PREP_neg) / (mean_PREN_neg - mean_PREP_neg)
n_theory_neg = quantile(law_g, target_prob_neg)

profit_theory_neg = mean(
    S_samples_neg .* n_theory_neg .- max.(
        PREP_samples_neg .* (n_theory_neg .- g_samples_neg), 
        PREN_samples_neg .* (n_theory_neg .- g_samples_neg)
    )
)

results_neg = DataFrame(eps = Float64[], n_opt = Float64[], profit = Float64[])
for eps in epsilons
    n_opt, cost = solve_dro_4D(g_samples_neg, S_samples_neg, PREP_samples_neg, PREN_samples_neg, eps, S_min_neg, S_max, PREP_min, PREP_max, PREN_min, PREN_max)    
    push!(results_neg, (eps = eps, n_opt = n_opt, profit = -cost))
end

# plots
p1 = plot(results_pos.eps, results_pos.profit, lw=2, marker=:+, legend=:topright, label="DRO-pos", xlabel=L"Radius $\epsilon$", ylabel="Profit")
plot!(p1, results_neg.eps, results_neg.profit, lw=2, marker=:+, label="DRO-neg")
hline!(p1, [profit_theory_pos], label="SAA", ls=:dash, color=:green, lw=2)

p2 = plot(results_pos.eps, results_pos.n_opt, lw=2, marker=:+, legend=:topright, label="DRO-pos", xlabel=L"Radius $\epsilon$", ylabel=L"Optimal nomination $n^{\star}$", ylims=(0.0, 1.0))
plot!(p2, results_neg.eps, results_neg.n_opt, lw=2, marker=:+, label="DRO-neg")

savefig(p1, "profit_sensitivity_WOB_stochastic_imbal.pdf")
savefig(p2, "decision_sensitivity_WOB_stochastic_imbal.pdf")

display(p1)
display(p2)
