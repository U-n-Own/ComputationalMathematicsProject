using LinearAlgebra
using IterativeSolvers
using SparseArrays
using Random
# Utilities
using NPZ
using Plots
using LightGraphs

include("GMRES.jl")

include("generate_matrix.jl")

Plots.default(show = true)

function load_mcfp_data(filename::String)
    """
    Load the MCFP data from the npz file

    Returns:
    D: Diagonal matrix
    E: Node arc incidence matrix
    edge_data: Full edge informations with a tuple of (source, target, least flow, max flow, cost)
    """

    data = NPZ.npzread(filename)

    flows = data["node_flows"]
    E = data["node_arc_matrix"]
    edge_data = data["full_edges"]

    return flows, E, edge_data
end

"""
PLOTS RESIDUAL OF GMRES
"""
function plot_residual_GMRES(D, E, A, y, iterations, step, title)
    iters = range(1, iterations, step = step)
    residuals = zeros(length(iters))

    for i in range(1, length(iters))
        if i % 20 == 0
            println(iters[i], "-th iteration")
        end

        x = GMRES_IncrementalQR(A, y, iters[i])
        residuals[i] = norm(A * x - y)

    end

    tic = 10.0 .^ collect(range(-12,12,step=2))

    p = plot(iters, residuals, yaxis=:log, yticks=tic, label = "residual", title = title)

end

"""
PLOTS RESIDUAL OF GMRES, PRECOND VS NON PRECOND
"""
function plot_residual_precond(D, E, A, y, iterations, step, title)
    iters = range(1, iterations, step = step)
    residuals = zeros(length(iters))

    for i in range(1, length(iters))
        if i % 20 == 0
            println(iters[i], "-th iteration")
        end

        x = GMRES_IncrementalQR(A, y, iters[i])
        residuals[i] = norm(A * x - y)

    end

    print("now precond")

    residuals_precond = zeros(length(iters))

    for i in range(1, length(iters))
        if i % 20 == 0
            println(iters[i], "-th iteration")
        end

        x = GMRES_IncrementalQR_precond(D, SparseMatrixCSC(E), y, iters[i])
        residuals_precond[i] = norm(A * x - y)

    end

    tic = 10.0 .^ collect(range(-12,12,step=2))

    p = plot(iters, [residuals residuals_precond], yaxis=:log, yticks=tic, label = ["no precond" "precond"], title ="All ones")

end


flows, E_bar, edge_data = load_mcfp_data("dataset/net10_8_1.dmx_out.npz")

D, E, A = gen_A_all_ones(E_bar)

y = (gen_y_from_data(flows, edge_data[:, 5]))

iterations = size(A)[1]

plot_residual_precond(D, E, A, y, iterations, 1000, "banana")

readline()
