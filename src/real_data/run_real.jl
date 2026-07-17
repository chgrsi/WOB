using JuMP
using Gurobi
using HiGHS
using Statistics
using DataFrames
using CSV
using Dates
using LinearAlgebra

const GRB_ENV = Gurobi.Env()

const K_VAL = 1.2
const TAU_VAL = (K_VAL - 1.0) / (K_VAL + 1.0)
const EPSILON = 10.0
const TRAIN_OBS = 30
const ALLOW_NEGATIVE_PRICES = true

# set uncertainty bounds (s_min, s_max, d_min, d_max) 
function support_bounds(s_tr, d_tr)
    s_max = 1.2*maximum(s_tr)
    d_max = 1.2*maximum(d_tr)
    s_min = -s_max
    d_min = -d_max
    if ALLOW_NEGATIVE_PRICES    
        return s_min, s_max, d_min, d_max
    else
        return max(0.0, s_min), s_max, max(0.0, d_min), d_max
    end
end

# parse timestamp and strip the suffix
function parse_timestamp_column(col)
    return DateTime.(
        replace.(string.(col), r"\+00:00$" => ""),
        dateformat"yyyy-mm-dd HH:MM:SS",
    )
end

# add timestamp, date, time cols
function add_time_columns!(df::DataFrame)
    df.timestamp = parse_timestamp_column(df[!, 1])
    df.date = Date.(df.timestamp)
    df.slot = Time.(df.timestamp)
    sort!(df, [:timestamp])
    return df
end

# rename r+/r- to rp/rm and ensure Float64, for true settlement profit
function add_settlement_columns!(df::DataFrame)
    Symbol("r+") in propertynames(df) && rename!(df, Symbol("r+") => :rp)
    Symbol("r-") in propertynames(df) && rename!(df, Symbol("r-") => :rm)
    df.rp = Float64.(df.rp)
    df.rm = Float64.(df.rm)
    return df
end

# keep clean rows only
function clean_training_data(data::DataFrame)
    clean = dropmissing(data, [:g, :s, :delta])
    filter!(row -> isfinite(row.g) && isfinite(row.s) && isfinite(row.delta), clean)
    return clean
end

# returns a Dict{Time, SubDataFrame} mapping each slot to its sorted observations
function build_slot_index(df::DataFrame)
    return Dict(
        key.slot => DataFrame(sdf)
        for (key, sdf) in pairs(groupby(df, :slot))
    )
end

# returns the last TRAIN_OBS valid observations for the given slot strictly
# before test_timestamp, or an empty DataFrame if insufficient data.
function get_training_data(
    slot_index::Dict,
    test_timestamp::DateTime,
    test_slot::Time,
)
    !haskey(slot_index, test_slot) && return DataFrame()

    slot_df = slot_index[test_slot]
    candidates = clean_training_data(
        slot_df[slot_df.timestamp .< test_timestamp, :]
    )

    nrow(candidates) < TRAIN_OBS && return DataFrame()

    return candidates[end-TRAIN_OBS+1:end, :]
end

# build support sets
function build_mccormick_support(s_min, s_max, d_min, d_max)
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
    d = [1.0, 0.0, s_max, -s_min, d_max, -d_min,
         0.0, s_max, -s_min, 0.0, 0.0, d_max, -d_min, 0.0]
    return C, d
end

function product_bounds(xL, xU, yL, yU)
    vals = (xL*yL, xL*yU, xU*yL, xU*yU)
    return minimum(vals), maximum(vals)
end

function build_box_support(s_min, s_max, d_min, d_max)
    gL, gU = 0.0, 1.0
    zsL, zsU = product_bounds(gL, gU, s_min, s_max)
    zdL, zdU = product_bounds(gL, gU, d_min, d_max)
    lower = [gL, s_min, d_min, zsL, zdL]
    upper = [gU, s_max, d_max, zsU, zdU]
    dim = 5
    C = zeros(2dim, dim); rhs = zeros(2dim)
    for j in 1:dim
        # xi_j <= upper_j
        C[2j-1, j] =  1.0; rhs[2j-1] =  upper[j]
        # -xi_j <= -lower_j
        C[2j, j] = -1.0; rhs[2j] = -lower[j]
    end
    return C, rhs
end

# profit
function obs_profit(n, g, s, rp, rm)
    return n * s + rm * max(g - n, 0.0) - rp * max(n - g, 0.0)
end

# solvers
function scaling_weights(g_tr, s_tr, d_tr)
    return [
        1 / max(std(g_tr), 1e-6),
        1 / max(std(s_tr), 1e-6),
        1 / max(std(d_tr), 1e-6),
        0.0,
        0.0,
    ]
end

function solve_saa(g_tr, s_tr, d_tr)
    N = length(g_tr)
    m = Model(HiGHS.Optimizer); set_silent(m)
    @variables(m, begin
        0 <= n <= 1
        loss[1:N]
    end)
    for i in 1:N
        cp = (s_tr[i] + d_tr[i]) * TAU_VAL - d_tr[i]
        cm = (s_tr[i] + d_tr[i]) * TAU_VAL + d_tr[i]
        @constraint(m, loss[i] >= -g_tr[i]*s_tr[i] + cp*(g_tr[i] - n))
        @constraint(m, loss[i] >= -g_tr[i]*s_tr[i] + cm*(n - g_tr[i]))
    end
    @objective(m, Min, sum(loss) / N)
    optimize!(m); return value(n)
end

function solve_ro(s_min, s_max, d_min, d_max)
    C, rhs = build_mccormick_support(s_min, s_max, d_min, d_max)
    K = size(C, 1)
    m = Model(HiGHS.Optimizer); set_silent(m)
    @variables(m, begin
        0 <= n <= 1
        wc
        g1[1:K] >= 0
        g2[1:K] >= 0
    end)
    B1 = [0.0, -TAU_VAL*n, (1.0-TAU_VAL)*n,  TAU_VAL-1.0,  TAU_VAL-1.0]
    B2 = [0.0,  TAU_VAL*n, (1.0+TAU_VAL)*n, -(1.0+TAU_VAL), -(1.0+TAU_VAL)]
    @constraint(m, C'*g1 .== B1)
    @constraint(m, dot(rhs, g1) <= wc)
    @constraint(m, C'*g2 .== B2)
    @constraint(m, dot(rhs, g2) <= wc)
    @objective(m, Min, wc)
    optimize!(m); return value(n)
end

function solve_dro_exact(g_tr, s_tr, d_tr, s_min, s_max, d_min, d_max, W)
    N = length(g_tr)
    m = Model(() -> Gurobi.Optimizer(GRB_ENV)); set_silent(m)
    @variables(m, begin
        0 <= n <= 1
        lam >= 0
        epi[1:N]
    end)
    for i in 1:N
        ξi = [g_tr[i], s_tr[i], d_tr[i], g_tr[i]*s_tr[i], g_tr[i]*d_tr[i]]
        for g_k in [0.0, g_tr[i], 1.0],
            s_k in [s_min, s_tr[i], s_max],
            d_k in [d_min, d_tr[i], d_max]
            ξk = [g_k, s_k, d_k, g_k*s_k, g_k*d_k]
            dist = sum(W[j] * abs(ξk[j] - ξi[j]) for j in 1:5)
            cp = (s_k + d_k) * TAU_VAL - d_k
            cm = (s_k + d_k) * TAU_VAL + d_k
            @constraint(m, epi[i] >= -g_k*s_k + cp*(g_k - n) - lam*dist)
            @constraint(m, epi[i] >= -g_k*s_k + cm*(n - g_k) - lam*dist)
        end
    end
    @objective(m, Min, lam*EPSILON + sum(epi)/N)
    optimize!(m); return value(n)
end

function solve_dro_mccormick(g_tr, s_tr, d_tr, s_min, s_max, d_min, d_max, W)
    N = length(g_tr)
    C, d = build_mccormick_support(s_min, s_max, d_min, d_max)
    K = size(C, 1)
    m = Model(HiGHS.Optimizer); set_silent(m)
    @variables(m, begin
        0 <= n <= 1; lam >= 0
        epi[1:N]
        g1[1:N, 1:K] >= 0
        g2[1:N, 1:K] >= 0
    end)
    B1 = [0.0, -TAU_VAL*n, (1.0-TAU_VAL)*n,  TAU_VAL-1.0,  TAU_VAL-1.0]
    B2 = [0.0,  TAU_VAL*n, (1.0+TAU_VAL)*n, -(1.0+TAU_VAL), -(1.0+TAU_VAL)]
    for i in 1:N
        ξi = [g_tr[i], s_tr[i], d_tr[i], g_tr[i]*s_tr[i], g_tr[i]*d_tr[i]]
        slack = d .- C * ξi
        cp = (s_tr[i] + d_tr[i]) * TAU_VAL - d_tr[i]
        cm = (s_tr[i] + d_tr[i]) * TAU_VAL + d_tr[i]
        @constraint(m, -g_tr[i]*s_tr[i] + cp*(g_tr[i]-n) + dot(g1[i,:], slack) <= epi[i])
        @constraint(m,  C'*g1[i,:] .- B1 .<= lam .* W)
        @constraint(m, -(C'*g1[i,:] .- B1) .<= lam .* W)
        @constraint(m, -g_tr[i]*s_tr[i] + cm*(n-g_tr[i]) + dot(g2[i,:], slack) <= epi[i])
        @constraint(m,  C'*g2[i,:] .- B2 .<= lam .* W)
        @constraint(m, -(C'*g2[i,:] .- B2) .<= lam .* W)
    end
    @objective(m, Min, lam*EPSILON + sum(epi)/N)
    optimize!(m); return value(n)
end

function solve_dro_box(g_tr, s_tr, d_tr, s_min, s_max, d_min, d_max, W)
    N = length(g_tr)
    C, rhs = build_box_support(s_min, s_max, d_min, d_max)
    K = size(C, 1)
    m = Model(HiGHS.Optimizer); set_silent(m)
    @variables(m, begin
        0 <= n <= 1; lam >= 0; epi[1:N]
        dual1[1:N, 1:K] >= 0; dual2[1:N, 1:K] >= 0
    end)
    B1 = [0.0, -TAU_VAL*n, (1.0-TAU_VAL)*n,  TAU_VAL-1.0,  TAU_VAL-1.0]
    B2 = [0.0,  TAU_VAL*n, (1.0+TAU_VAL)*n, -(1.0+TAU_VAL), -(1.0+TAU_VAL)]
    for i in 1:N
        ξi = [g_tr[i], s_tr[i], d_tr[i], g_tr[i]*s_tr[i], g_tr[i]*d_tr[i]]
        slack = rhs .- C * ξi
        @constraint(m, dot(B1, ξi) + dot(dual1[i,:], slack) <= epi[i])
        @constraint(m,  C'*dual1[i,:] .- B1 .<= lam .* W)
        @constraint(m, -(C'*dual1[i,:] .- B1) .<= lam .* W)
        @constraint(m, dot(B2, ξi) + dot(dual2[i,:], slack) <= epi[i])
        @constraint(m,  C'*dual2[i,:] .- B2 .<= lam .* W)
        @constraint(m, -(C'*dual2[i,:] .- B2) .<= lam .* W)
    end
    @objective(m, Min, lam*EPSILON + sum(epi)/N)
    optimize!(m); return value(n)
end

# backtest
function run_real(input_file::String, output_base::String, test_date::Union{Date, Nothing}=nothing)
    df = CSV.read(input_file, DataFrame)
    add_time_columns!(df)
    add_settlement_columns!(df)

    if !isnothing(test_date)
        df_test = filter(row -> row.date == test_date, df)
        date_str = Dates.format(test_date, "yyyy-mm-dd")
        output_dir = joinpath(output_base, date_str)
        if nrow(df_test) == 0
            println("No data found for date $test_date")
            return
        end
    else
        df_test = df
        output_dir = output_base
    end

    println()
    println("Input:  $input_file")
    println("Output: $output_dir")
    println("Period: $(minimum(df.timestamp)) → $(maximum(df.timestamp))")
    println("Slots per day: $(length(unique(df.slot)))")
    println("Training: last $TRAIN_OBS valid observations per slot")
    println("Negative prices in support: $ALLOW_NEGATIVE_PRICES")

    # pre-group by slot 
    slot_index = build_slot_index(df)

    results = DataFrame(
        timestamp        = DateTime[],
        date             = Date[],
        slot             = Time[],
        n_saa            = Float64[],
        n_ro             = Float64[],
        n_dro_exact      = Float64[],
        n_dro_mccormick  = Float64[],
        n_dro_box        = Float64[],
        profit_saa       = Float64[],
        profit_ro        = Float64[],
        profit_dro_exact = Float64[],
        profit_dro_mcco  = Float64[],
        profit_dro_box   = Float64[],
    )

    n_total = nrow(df_test)
    for (i, test_row) in enumerate(eachrow(df_test))
        train = get_training_data(slot_index, test_row.timestamp, test_row.slot)
        nrow(train) < TRAIN_OBS && continue

        g_tr, s_tr, d_tr = train.g, train.s, train.delta
        s_min, s_max, d_min, d_max = support_bounds(s_tr, d_tr)
        W = scaling_weights(g_tr, s_tr, d_tr)

        n_saa = solve_saa(g_tr, s_tr, d_tr)
        n_ro = solve_ro(s_min, s_max, d_min, d_max)
        n_exact = solve_dro_exact(g_tr, s_tr, d_tr, s_min, s_max, d_min, d_max, W)
        n_mccormick = solve_dro_mccormick(g_tr, s_tr, d_tr, s_min, s_max, d_min, d_max, W)
        n_box = solve_dro_box(g_tr, s_tr, d_tr, s_min, s_max, d_min, d_max, W)

        g, s = test_row.g, test_row.s
        rp, rm = test_row.rp, test_row.rm

        push!(results, (
            test_row.timestamp, test_row.date, test_row.slot,
            n_saa, n_ro, n_exact, n_mccormick, n_box,
            obs_profit(n_saa, g, s, rp, rm),
            obs_profit(n_ro, g, s, rp, rm),
            obs_profit(n_exact, g, s, rp, rm),
            obs_profit(n_mccormick, g, s, rp, rm),
            obs_profit(n_box, g, s, rp, rm),
        ))

        i % 100 == 0 && println("  $i / $n_total")
    end

    mkpath(output_dir)
    out_path = joinpath(output_dir, "oos_profits.csv")
    CSV.write(out_path, results)
    println("Saved $out_path  ($(nrow(results)) rows)")
end

# run_real("real_data_15min.csv",  "results/real/15min")
run_real("real_data_hourly.csv", "results/real/hourly")
