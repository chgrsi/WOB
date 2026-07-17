using JuMP
using HiGHS
using Gurobi
using LinearAlgebra
using Statistics
using StatsBase  

const GRB_ENV = Gurobi.Env()

# Compute loss
function loss(n, g, s, d, tau)
    cm = (s + d) * tau - d   # c- (surplus)
    cp = (s + d) * tau + d   # c+ (deficit)
    return -g * s + cm * max(g - n, 0.0) + cp * max(n - g, 0.0)
end

# Profit is the opposite of the loss
profit(n, g, s, d, tau) = -loss(n, g, s, d, tau)

# Compute out-of-sample metrics
function compute_oos_metrics(n, g, s, d, tau; alpha = 0.001, clairvoyant = nothing)
    profits = profit.(n, g, s, d, tau)
    sorted_profits = sort(profits)
    n_tail = max(1, round(Int, alpha * length(profits)))
    sigma = std(profits)
    downside_deviation = sqrt(mean(min.(profits, 0.0) .^ 2)) 
    if clairvoyant == nothing
        clairvoyant = max.(profit.(0.0, g, s, d, tau),
                        profit.(g, g, s, d, tau),
                        profit.(1.0, g, s, d, tau))
    end
    regret = clairvoyant .- profits
    deficit = g .< n  # short  
    surplus = g .> n  # long   
    return (mean = mean(profits),
            cvar = mean(sorted_profits[1:n_tail]),
            sharpe = sigma > 0 ? mean(profits) / sigma : NaN,
            sortino = downside_deviation > 0 ? mean(profits) / downside_deviation : NaN,
            neg_freq = mean(profits .< 0.0),
            regret_mean = mean(regret),
            regret_q95 = quantile(regret, 0.95),
            profit_deficit = any(deficit) ? mean(profits[deficit]) : NaN,
            profit_surplus = any(surplus) ? mean(profits[surplus]) : NaN)
    end

# SAA solving the LP
function solve_saa(g_train, s_train, d_train, tau)
    N = length(g_train)
    n_bad = count(s_train .+ d_train .< 0.0)
    n_bad == 0 || @warn "$n_bad/$N training samples have s+δ < 0"
    
    m = Model(HiGHS.Optimizer); set_silent(m)
    
    @variables(m, begin 
        0 <= n <= 1
        loss[1:N] 
    end)
    
    for i in 1:N
        # cm = surplus price, cp = deficit price
        cm = (s_train[i] + d_train[i]) * tau - d_train[i]
        cp = (s_train[i] + d_train[i]) * tau + d_train[i]
        # surplus (g > n)
        @constraint(m, loss[i] >= -g_train[i]*s_train[i] + cm*(g_train[i] - n))
        # deficit (n > g)
        @constraint(m, loss[i] >= -g_train[i]*s_train[i] + cp*(n - g_train[i]))
    end
    # minimize the average loss over historical samples
    @objective(m, Min, sum(loss) / N)
    optimize!(m)
    if termination_status(m) != MOI.OPTIMAL
        @warn "Status: $(termination_status(m))"
        return NaN
    end
    return value(n)
end

# SAA exact as the argmin of the average loss
function solve_saa_exact(g_train, s_train, d_train, tau)
    candidates = unique!(sort!(vcat(0.0, 1.0, Float64.(g_train))))
    avg_loss(n) = mean(loss(n, g_train[i], s_train[i], d_train[i], tau)
                       for i in eachindex(g_train))
    return candidates[argmin(avg_loss.(candidates))]
end

# Lifted supports
function product_bounds(xL, xU, yL, yU)
    corners = (xL * yL, xL * yU, xU * yL, xU * yU)
    return minimum(corners), maximum(corners)
end
 
# McCormick envelopes of t_s = gs and t_δ = gδ over the box
function mccormick_support(Xi)
    s_min, s_max = Xi.s_min, Xi.s_max
    d_min, d_max = Xi.d_min, Xi.d_max
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
 
# Coordinate-wise bounds 
function box_support(Xi)
    ts_min, ts_max = product_bounds(0.0, 1.0, Xi.s_min, Xi.s_max)
    td_min, td_max = product_bounds(0.0, 1.0, Xi.d_min, Xi.d_max)
    
    lower = [0.0, Xi.s_min, Xi.d_min, ts_min, td_min]
    upper = [1.0, Xi.s_max, Xi.d_max, ts_max, td_max]
    
    C = zeros(10, 5)
    d = zeros(10)
    for j in 1:5
        C[2j - 1, j] = 1.0
        d[2j - 1] = upper[j]
        C[2j, j] = -1.0
        d[2j] = -lower[j]
    end
    return C, d
end

# RO over a relaxed set 
function solve_ro_approx(C, d, tau)
    K = size(C, 1)
    model = Model(HiGHS.Optimizer); set_silent(model)

    @variable(model, 0 <= n <= 1)
    @variable(model, worst_case)
    @variable(model, gam1[1:K] >= 0) # duality multipliers (surplus) 
    @variable(model, gam2[1:K] >= 0) # duality multipliers (deficit) 
    
    # Lifted loss coefficients of each branch
    B1 = [0.0, -tau * n, (1.0 - tau) * n, tau - 1.0, tau - 1.0]
    B2 = [0.0,  tau * n, (1.0 + tau) * n, -(1.0 + tau), -(1.0 + tau)]
    
    @constraint(model, C' * gam1 .== B1)
    @constraint(model, dot(d, gam1) <= worst_case)
    @constraint(model, C' * gam2 .== B2)
    @constraint(model, dot(d, gam2) <= worst_case)
    
    # minimize worst-case lifted loss over the relaxed box {ξ̃ : Cξ̃ ≤ d}
    @objective(model, Min, worst_case)
    optimize!(model)
    if termination_status(model) != MOI.OPTIMAL
        @warn "Status: $(termination_status(model))"
        return NaN
    end
    return value(n)
end


function solve_ro_mccormick(Xi, tau)
    C, d = mccormick_support(Xi)
    return solve_ro_approx(C, d, tau)
end
 
function solve_ro_box(Xi, tau)
    C, d = box_support(Xi)
    return solve_ro_approx(C, d, tau)
end


# Generate candidate points
function candidate_pieces(n, ghat, shat, dhat, Xi, W, tau)
    values = Float64[]
    costs = Float64[]
    for g in (0.0, ghat, 1.0, n)
        for s in (Xi.s_min, shat, Xi.s_max)
            for d in (Xi.d_min, dhat, Xi.d_max)
                push!(values, loss(n, g, s, d, tau))
                push!(costs, W[1] * abs(g - ghat) + W[2] * abs(s - shat) + W[3] * abs(d - dhat))
            end
        end
    end
    return values, costs
end


# exact DRO over a grid of epsilons
function solve_dro_exact_grid(g_tr, s_tr, d_tr, Xi, W, tau, epsilons;
                               ngrid::Int = 1001)
    N = length(g_tr)
    n_grid = unique!(sort!(vcat(collect(range(0.0, 1.0; length = ngrid)),
                                Float64.(g_tr))))
 
    best_value = fill(Inf, length(epsilons))
    best_n = zeros(length(epsilons))
    best_lam = zeros(length(epsilons))
 
    for n in n_grid
        model = Model(() -> Gurobi.Optimizer(GRB_ENV)); set_silent(model)
        @variable(model, lam >= 0)
        @variable(model, epi[1:N])
        for i in 1:N
            values, costs = candidate_pieces(n, g_tr[i], s_tr[i], d_tr[i], Xi, W, tau)
            for p in eachindex(values)
                @constraint(model, epi[i] >= values[p] - lam * costs[p])
            end
        end
        for j in eachindex(epsilons)
            @objective(model, Min, lam * epsilons[j] + sum(epi) / N)
            optimize!(model)
            if termination_status(model) != MOI.OPTIMAL
                @warn "Solver failed for n=$n, epsilon=$(epsilons[j]). Status: $(termination_status(model))"
                continue
            end
            if objective_value(model) < best_value[j]
                best_value[j] = objective_value(model)
                best_n[j] = n
                best_lam[j] = value(lam)
            end
        end
    end
    return best_n, -best_value, best_lam
end


# Exact DRO for a single epsilon
function solve_dro_exact(g_tr, s_tr, d_tr, Xi, W, tau, eps; ngrid::Int = 1001)
    ns, profits, lams = solve_dro_exact_grid(g_tr, s_tr, d_tr, Xi, W, tau, [eps]; ngrid = ngrid)
    return ns[1], profits[1], lams[1]
end


# Conservative relaxation (Esfahani–Kuhn)
function build_relaxation_model(C, d, g_tr, s_tr, d_tr, W, tau)
    N = length(g_tr)
    K = size(C, 1)
    W5 = [W[1], W[2], W[3], 0.0, 0.0]
 
    model = Model(() -> HiGHS.Optimizer()); set_silent(model)
    @variable(model, 0 <= n <= 1)
    @variable(model, lam >= 0)
    @variable(model, epi[1:N])
    @variable(model, gam1[1:N, 1:K] >= 0) # surplus branch multipliers
    @variable(model, gam2[1:N, 1:K] >= 0) # deficit branch multipliers
    
    # lifted loss coefficients of each branch (affine in n)
    B1 = [0.0, -tau * n, (1.0 - tau) * n, tau - 1.0, tau - 1.0] # surplus
    B2 = [0.0,  tau * n, (1.0 + tau) * n, -(1.0 + tau), -(1.0 + tau)] # deficit
 
    for i in 1:N
        xi = [g_tr[i], s_tr[i], d_tr[i], g_tr[i] * s_tr[i], g_tr[i] * d_tr[i]]
        slack = d .- C * xi
        cm = (s_tr[i] + d_tr[i]) * tau - d_tr[i]   # c-
        cp = (s_tr[i] + d_tr[i]) * tau + d_tr[i]   # c+
 
        @constraint(model, -g_tr[i] * s_tr[i] + cm * (g_tr[i] - n) + dot(gam1[i, :], slack) <= epi[i])
        @constraint(model,   C' * gam1[i, :] .- B1 .<= lam .* W5)
        @constraint(model, -(C' * gam1[i, :] .- B1) .<= lam .* W5)
 
        @constraint(model, -g_tr[i] * s_tr[i] + cp * (n - g_tr[i]) + dot(gam2[i, :], slack) <= epi[i])
        @constraint(model,   C' * gam2[i, :] .- B2 .<= lam .* W5)
        @constraint(model, -(C' * gam2[i, :] .- B2) .<= lam .* W5)
    end
    return model, n, lam, epi
end


# Solve relaxation LP over a grid of epsilons
function solve_relaxations_over_epsilons(model, n, lam, epi, N, epsilons)
    n_opts = Float64[]
    insample_profits = Float64[]
    lam_opts = Float64[]
    for eps in epsilons
        @objective(model, Min, lam * eps + sum(epi) / N)
        optimize!(model)
        if termination_status(model) != MOI.OPTIMAL
            @warn "Solver failed for epsilon=$eps. Status: $(termination_status(model))"
            push!(n_opts, NaN)
            push!(insample_profits, NaN)
            push!(lam_opts, NaN)
            continue
        end
        
        push!(n_opts, value(n))
        push!(insample_profits, -objective_value(model))
        push!(lam_opts, value(lam))
    end
    return n_opts, insample_profits, lam_opts
end

function solve_dro_approx_mccormick_grid(g_tr, s_tr, d_tr, Xi, W, tau, epsilons)
    C, d = mccormick_support(Xi)
    model, n, lam, epi = build_relaxation_model(C, d, g_tr, s_tr, d_tr, W, tau)
    return solve_relaxations_over_epsilons(model, n, lam, epi, length(g_tr), epsilons)
end

function solve_dro_approx_box_grid(g_tr, s_tr, d_tr, Xi, W, tau, epsilons)
    C, d = box_support(Xi)
    model, n, lam, epi = build_relaxation_model(C, d, g_tr, s_tr, d_tr, W, tau)
    return solve_relaxations_over_epsilons(model, n, lam, epi, length(g_tr), epsilons)
end
 
function solve_dro_approx_mccormick(g_tr, s_tr, d_tr, Xi, W, tau, eps)
    ns, profits, lams = solve_dro_approx_mccormick_grid(g_tr, s_tr, d_tr, Xi, W, tau, [eps])
    return ns[1], profits[1], lams[1]
end
 
function solve_dro_approx_box(g_tr, s_tr, d_tr, Xi, W, tau, eps)
    ns, profits, lams = solve_dro_approx_box_grid(g_tr, s_tr, d_tr, Xi, W, tau, [eps])
    return ns[1], profits[1], lams[1]
end

# Data-driven weights
# MAD and IQR are robust to rare spikes contrary to std
function dispersion(x, method::Symbol)
    method == :mad && return mad(x; normalize = true)
    method == :iqr && return iqr(x) / 1.349
    method == :std && return std(x)
    error("unknown dispersion method: $method")
end
 
function scaling_weights(g_tr, s_tr, d_tr; mult = (g = 1.0, s = 1.0, d = 1.0), method::Symbol = :mad)
    return [mult.g / max(dispersion(g_tr, method), 1e-6),
            mult.s / max(dispersion(s_tr, method), 1e-6),
            mult.d / max(dispersion(d_tr, method), 1e-6)]
end

# Check that the no arbitrage condition is verified
function check_no_arbitrage(s, d, tau)
    r = s .+ d
    E_rm = mean((1 - tau) .* r)
    E_s = mean(s)
    E_rp = mean((1 + tau) .* r)
    println("E[r⁻] = ", round(E_rm, digits = 1),
            "   E[s] = ", round(E_s, digits = 1),
            "   E[r⁺] = ", round(E_rp, digits = 1))
    E_rm < E_s || @warn "arbitrage: average surplus price exceeds day-ahead price"
    E_rp > E_s || @warn "arbitrage: average deficit price below day-ahead price"
    return nothing
end
