using Plots
using DataFrames
using CSV
using LaTeXStrings

default(
    fontfamily = "Computer Modern", titlefontsize = 14, guidefontsize = 12,             
    tickfontsize = 10, legendfontsize = 8, framestyle = :box,             
    grid = true, gridalpha = 0.3, margin = 5Plots.mm              
)

function generate_plots()
    mkpath("plots")
    files = filter(f -> endswith(f, ".csv"), readdir("results/raw", join=true))
    mkpath("plots/raw")

    colors = Dict(
        ("pos", "exact")  => :dodgerblue,
        ("neg", "exact")  => :indianred1,
        ("pos", "approx") => :forestgreen,
        ("neg", "approx") => :darkorange
    )
    
    line_width = 3
    my_xticks = 10.0 .^ (-4:1:4)

    for file in files
        df = CSV.read(file, DataFrame)
        N_train = match(r"N_(\d+)", file).captures[1]
        
        # profit 
        p1 = plot(xscale=:log10, xticks=my_xticks, xminorgrid=true, minorgridalpha=0.15, 
                  xlabel=L"Radius $\epsilon$", ylabel="Profit [EUR/MWh]", legend=:bottomleft)

        # reliability
        p1_twin = twinx(p1)
        plot!(p1_twin, xscale=:log10, xticks=my_xticks, ylabel="Reliability", 
              ylims=(0.0, 1.05), grid=false, legend=:bottomright)

        # decision
        p2 = plot(xscale=:log10, xticks=my_xticks, xminorgrid=true, minorgridalpha=0.15, 
                  xlabel=L"Radius $\epsilon$", ylabel=L"Optimal $n^{\star}$", ylims=(0.0, 1.0), legend=:topleft)
        
        # CVaR
        p3 = plot(xscale=:log10, xticks=my_xticks, xminorgrid=true, minorgridalpha=0.15, 
                  xlabel=L"Radius $\epsilon$", ylabel="5% Tail Risk [EUR/MWh]", legend=:topleft)

        # lambda
        p4 = plot(xscale=:log10, yscale=:log10, xticks=my_xticks, xminorgrid=true, 
          xlabel=L"Radius $\epsilon$", ylabel=L"Dual variable $\lambda$", legend=:topright)

        for scen in unique(df.Scenario)
            for meth in unique(df.Method)
                sub = filter(r -> r.Scenario == scen && r.Method == meth, df)
                lbl = "$meth ($scen)"
                col = colors[(scen, meth)]

                # profit
                plot!(p1, sub.Epsilon, sub.Profit_Mean, 
                      ribbon=(sub.Profit_Mean .- sub.Profit_Q20, sub.Profit_Q80 .- sub.Profit_Mean),
                      fillalpha=0.15, color=col, lw=line_width, label=lbl)
                # reliability
                plot!(p1_twin, sub.Epsilon, sub.Reliability, xscale=:log10, 
                      color=col, lw=line_width, ls=:dashdot, label="Rel. "*lbl)
                # decision
                plot!(p2, sub.Epsilon, sub.N_opt, color=col, lw=line_width, label=lbl)
                # CVaR
                plot!(p3, sub.Epsilon, sub.CVaR5, color=col, lw=line_width, label=lbl)
                # lambda
                plot!(p4, sub.Epsilon, max.(sub.Lambda, 1e-8), color=col, lw=line_width, label="$meth ($scen)")
            end
            
            sub_b = filter(r -> r.Scenario == scen && r.Method == "exact", df)
            base_col = colors[(scen, "exact")]
            
            hline!(p1, [sub_b.SAA_Profit[1]], color=base_col, ls=:dash, lw=line_width, label="SAA ($scen)")
            hline!(p1, [sub_b.RO_Profit[1]], color=base_col, ls=:dot, lw=line_width, label="RO ($scen)")
            
            hline!(p2, [sub_b.SAA_N[1]], color=base_col, ls=:dash, lw=line_width, label="SAA ($scen)")
            hline!(p2, [sub_b.RO_N[1]], color=base_col, ls=:dot, lw=line_width, label="RO ($scen)")
            
            hline!(p3, [sub_b.SAA_CVaR[1]], color=base_col, ls=:dash, lw=line_width, label="SAA ($scen)")
            hline!(p3, [sub_b.RO_CVaR[1]], color=base_col, ls=:dot, lw=line_width, label="RO ($scen)")
        end

        savefig(p1, "plots/raw/profit_N_$(N_train).pdf")
        savefig(p2, "plots/raw/decision_N_$(N_train).pdf")
        savefig(p3, "plots/raw/cvar_N_$(N_train).pdf")
        savefig(p4, "plots/raw/lambda_N_$(N_train).pdf")
        println("Generated plots for N_train = $N_train")
    end
end

generate_plots()
