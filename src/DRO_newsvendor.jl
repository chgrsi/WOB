using JuMP, HiGHS, DataFrames, Distributions, Plots, LinearAlgebra, LaTeXStrings
using FastGaussQuadrature

# style setup 
default(
    fontfamily = "Computer Modern", 
    titlefontsize = 18,
    guidefontsize = 18,             
    tickfontsize = 16,             
    legendfontsize = 16,          
    linewidth = 2,                  
    framestyle = :box,             
    grid = true,
    gridalpha = 0.3,              
    margin = 5Plots.mm              
)


function gaussian_quadrature(law::Normal, N::Int)  
    x0, w0 = FastGaussQuadrature.gausshermite(N)  
    x = (sqrt(2.0) * law.σ) .* x0 .+ law.μ  
    w = w0 ./ sqrt(pi)  
    return (x, w)  
end

function solve_dro_newsvendor(g_data::Vector{Float64}, w::Vector{Float64}, epsilon::Float64, S::Float64, PREP::Float64, PREN::Float64)
    N = length(g_data)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    # Support constraints for g: [0, 1] -> C*g <= d
    C = [1.0, -1.0] 
    d = [1.0, 0.0]
    
    @variables(model, begin
        0 <= n <= 1            # decision variable
        lambda >= 0            # Wasserstein dual variable
        s[1:N]                 # epigraph variables
        gamma1[1:N, 1:2] >= 0  # duals for surplus 
        gamma2[1:N, 1:2] >= 0  # duals for deficit 
    end)
    
    for i in 1:N
        g_i = g_data[i]
        slack = d .- C .* g_i
        
        # Surplus
        loss1 = -S * n + PREP * (n - g_i)
        B1 = -PREP
        @constraint(model, loss1 + dot(gamma1[i, :], slack) <= s[i])
        @constraint(model, C' * gamma1[i, :] - B1 <= lambda)
        @constraint(model, -(C' * gamma1[i, :] - B1) <= lambda)
        
        # Deficit 
        loss2 = -S * n + PREN * (n - g_i)
        B2 = -PREN
        @constraint(model, loss2 + dot(gamma2[i, :], slack) <= s[i])
        @constraint(model, C' * gamma2[i, :] - B2 <= lambda)
        @constraint(model, -(C' * gamma2[i, :] - B2) <= lambda)
    end
    
    @objective(model, Min, lambda * epsilon + sum(w[i] * s[i] for i in 1:N))
    optimize!(model)
    
    return value(n), objective_value(model)
end

# params
mu_g, sigma_g = 0.7, 0.1
S_det, PREP_det, PREN_det = 65.0, 55.0, 75.0 
N_samples = 1000
law = Normal(mu_g, sigma_g)
g_nodes_raw, g_weights = gaussian_quadrature(law, 15) 
g_nodes = clamp.(g_nodes_raw, 0.0, 1.0)

# Newsvendor theoretical baseline
target_prob = (S_det - PREP_det) / (PREN_det - PREP_det)
n_theory = quantile(law, target_prob)
profit_theory = sum(g_weights .* (S_det * n_theory .- max.(PREP_det .* (n_theory .- g_nodes), PREN_det .* (n_theory .- g_nodes))))

# analysis
epsilons = 0.0:0.01:1.0
results = DataFrame(eps = Float64[], n_opt = Float64[], profit = Float64[])

for eps in epsilons
    n_opt, cost = solve_dro_newsvendor(g_nodes, g_weights, eps, S_det, PREP_det, PREN_det)
    push!(results, (eps, n_opt, -cost))
end

p1 = plot(results.eps, results.profit, lw=2, marker=:+, legend=:topright, label="DRO", xlabel=L"Radius $\epsilon$", ylabel="Profit")
p2 = plot(results.eps, results.n_opt, lw=2, marker=:+, legend=:topright, label="DRO", xlabel=L"Radius $\epsilon$", ylabel=L"Optimal nomination $n^{\star}$", ylims=(0.0, 1.0))
hline!(p1, [profit_theory], label="SAA", ls=:dash, color=:green, lw=2)
hline!(p2, [n_theory], label="SAA", ls=:dash, color=:green, lw=2)

savefig(p1, "profit_sensitivity_DRO_newsvendor.pdf")
savefig(p2, "decision_sensitivity_DRO_newsvendor.pdf")
