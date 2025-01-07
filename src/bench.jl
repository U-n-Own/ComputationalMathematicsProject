using LinearAlgebra
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

    p = plot(iters, [residuals residuals_precond], yaxis=:log, yticks=tic, label = ["no precond" "precond"], title = title)

end

"""
GENERATES SCATTER PLOTS COMPARING ALL METHODS
"""
function plot_residual_restarted(D, E, A, y, iterations, step, title)
    iters = range(1, iterations, step = step)
    times = zeros(length(iters))
    times_precond = zeros(length(iters))
    times_restarted = zeros(length(iters))
    times_restarted_precond = zeros(length(iters))
    residuals = zeros(length(iters))
    residuals_precond = zeros(length(iters))
    residuals_restarted = zeros(length(iters))
    residuals_restarted_precond = zeros(length(iters))

    for i in range(1, length(iters))
        if i % 20 == 0
            println(iters[i], "-th iteration")
        end

        t = @elapsed begin
            x = GMRES_IncrementalQR(A, y, iters[i])
        end
        residuals[i] = norm(A * x - y)
        times[i] = t
    end

     for i in range(1, length(iters))
        if i % 20 == 0
            println(iters[i], "-th iteration")
        end

        t = @elapsed begin
            x = GMRES_IncrementalQR_precond(D, SparseMatrixCSC(E), y, iters[i])
        end
        residuals_precond[i] = norm(A * x - y)
        times_precond[i] = t
    end

    for i in range(1, length(iters))
        if i % 20 == 0
            println(iters[i], "-th iteration")
        end

        t = @elapsed begin
            x = GMRES_Restarted(GMRES_IncrementalQR, A, y, (A,), iters[i], 2000, 10E-10)
        end
        residuals_restarted[i] = norm(A * x - y)
        times_restarted[i] = t
    end

    for i in range(1, length(iters))
        if i % 20 == 0
            println(iters[i], "-th iteration")
        end

        t = @elapsed begin
            x = GMRES_Restarted(GMRES_IncrementalQR_precond, A, y, (D, SparseMatrixCSC(E)), iters[i], 300, 10E-10)
        end
        residuals_restarted_precond[i] = norm(A * x - y)
        times_restarted_precond[i] = t
    end

    tic = 10.0 .^ collect(range(-12,12,step=2))

#     argmax = findmax(times)[2]
#     deleteat!(times, argmax)
#     deleteat!(residuals, argmax)

    p = scatter([times, times_precond, times_restarted, times_restarted_precond],
                [residuals, residuals_precond, residuals_restarted, residuals_restarted_precond], yaxis=:log, yticks=tic, label = ["GMRES" "Precond" "Restarted" "Restarted Precond"], title = title)

end
