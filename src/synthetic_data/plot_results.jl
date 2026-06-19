using Plots
using DataFrames
using CSV
using LaTeXStrings

default(
    fontfamily = "Computer Modern",
    titlefontsize = 14,
    guidefontsize = 12,
    tickfontsize = 10,
    legendfontsize = 8,
    framestyle = :box,
    grid = true,
    gridalpha = 0.3,
    margin = 5Plots.mm
)

function generate_plots()
    results_dir = "results/raw"
    plots_dir = "plots/raw"
    mkpath(plots_dir)

    files = filter(f -> endswith(f, ".csv"), readdir(results_dir, join = true))

    colors = Dict(
        "approx-box" => :chocolate1,
        "approx-mccormick" => :chocolate4,
        "exact" => :black,

    )

    line_width = 3
    my_xticks = 10.0 .^ (-4:1:4)

    for file in files 
        df = CSV.read(file, DataFrame)
        N_train = match(r"N_(\d+)", file).captures[1]

        for scen in ["pos", "neg"]

            # Profit
            p1 = plot( xscale = :log10, xticks = my_xticks, xminorgrid = true, minorgridalpha = 0.15,
                xlabel = L"Radius $\epsilon$", ylabel = "Profit [EUR/MWh]", legend = :bottomleft,
            )

            # Reliability twin axis
            p1_twin = twinx(p1)
            plot!(p1_twin, xscale = :log10, xticks = my_xticks, ylabel = "Reliability", ylims = (0.0, 1.05), grid = false, legend = :bottomright,)

            # Decision
            p2 = plot( xscale = :log10, xticks = my_xticks, xminorgrid = true, minorgridalpha = 0.15,
                xlabel = L"Radius $\epsilon$", ylabel = L"Optimal $n^{\star}$", ylims = (0.0, 1.0), legend = :topleft,
            )

            # CVaR
            p3 = plot(xscale = :log10, xticks = my_xticks, xminorgrid = true, minorgridalpha = 0.15,
                xlabel = L"Radius $\epsilon$", ylabel = "5% Tail Risk [EUR/MWh]", legend = :topleft,
            )

            # Lambda
            p4 = plot( xscale = :log10, yscale = :log10, xticks = my_xticks, xminorgrid = true, minorgridalpha = 0.15,
                xlabel = L"Radius $\epsilon$", ylabel = L"Dual variable $\lambda$", legend = :topright,
            )

            for meth in ["exact", "approx-mccormick", "approx-box"]
                sub = filter(r -> r.Scenario == scen && r.Method == meth, df)
                sort!(sub, :Epsilon)

                lbl = meth
                col = colors[meth]

                # Profit
                plot!(
                    p1, 
                    sub.Epsilon, 
                    sub.Profit_Mean,
                    ribbon = (
                        sub.Profit_Mean .- sub.Profit_Q20,
                        sub.Profit_Q80 .- sub.Profit_Mean,
                    ),
                    fillalpha = 0.15,
                    color = col,
                    lw = line_width,
                    label = lbl,
                )

                # Reliability
                plot!(p1_twin, sub.Epsilon, sub.Reliability, xscale = :log10, color = col, lw = line_width, ls = :dashdot, label = false,)

                # Decision
                plot!(p2, sub.Epsilon, sub.N_opt, color = col, lw = line_width, label = lbl,)

                # CVaR
                plot!(p3, sub.Epsilon, sub.CVaR5, color = col, lw = line_width, label = lbl,)

                # Lambda
                plot!(p4, sub.Epsilon, max.(sub.Lambda, 1e-8), color = col, lw = line_width, label = lbl,)
            end

            # approx-mccormick
            sub_b = filter(r -> r.Scenario == scen && r.Method == "approx-mccormick", df)

            hline!(p1, [sub_b.SAA_Profit[1]], color = :dodgerblue, ls = :dash, lw = line_width, label = "SAA",)
            hline!(p1, [sub_b.RO_Profit[1]], color = :indianred1, ls = :dash, lw = line_width, label = "RO",)
            hline!(p2, [sub_b.SAA_N[1]], color = :dodgerblue, ls = :dash, lw = line_width, label = "SAA",)
            hline!(p2, [sub_b.RO_N[1]], color = :indianred1, ls = :dash, lw = line_width, label = "RO",)
            hline!(p3, [sub_b.SAA_CVaR[1]], color = :dodgerblue, ls = :dash, lw = line_width, label = "SAA",)
            hline!(p3, [sub_b.RO_CVaR[1]], color = :indianred1, ls = :dash, lw = line_width, label = "RO",)

            savefig(p1, joinpath(plots_dir, "profit_$(scen)_N_$(N_train).pdf"))
            savefig(p2, joinpath(plots_dir, "decision_$(scen)_N_$(N_train).pdf"))
            savefig(p3, joinpath(plots_dir, "cvar_$(scen)_N_$(N_train).pdf"))
            savefig(p4, joinpath(plots_dir, "lambda_$(scen)_N_$(N_train).pdf"))
        end
    end
end

generate_plots()
