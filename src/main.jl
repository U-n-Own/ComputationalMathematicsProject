using ArgParse
using LinearAlgebra
using SparseArrays
using Random
# Utilities
using NPZ
using Plots
using LightGraphs
using JLD

include("GMRES.jl")
include("generate_matrix.jl")
include("bench.jl")

"""
Converts numpy's CSC format to Julia's
"""
function pycsc_to_csc(py_csc)
    size = py_csc.shape
    res = spzeros(size[1], size[2])

    println("converting data format...")

    t = @elapsed for j in 1:size[2]
            nz = py_csc.getcol(j-1).nonzero()[1]
            res[nz[1] + 1, j] = py_csc[nz[1] + 1, j]
            res[nz[2] + 1, j] = py_csc[nz[2] + 1, j]
    end

    println("converted in ", t, " seconds.\n")

    return res
end

function load_mcfp_data(filename::String)
    """
    Load the MCFP data from the npz file

    Returns:
    D: Diagonal matrix
    E: Node arc incidence matrix
    edge_data: Full edge informations with a tuple of (source, target, least flow, max flow, cost)
    """
    np = pyimport("numpy")

    data = np.load(filename, allow_pickle=true)

    flows = data.get("node_flows")

    E = pycsc_to_csc(data.get("node_arc_matrix")[])

    edge_data = data.get("full_edges")

    return flows, E, edge_data
end


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--iterations"
            help = "Number of iterations."
            arg_type = Int
            default = 1000
        "--restart"
            help = "Restart parameter of restarted algorithm."
            arg_type = Int
            default = 300
        "--restart_precond"
            help = "Restart parameter of restarted preconditioned algorithm."
            arg_type = Int
            default = 300
        "--diagonal"
            help = "Can be \"ones\", \"uniform\" or \"quadratic\""
            default = "quadratic"
        "--tol"
            help = "Tolerance parameter of restarted algorithm."
            arg_type = Float64
            default = 10e-10
        "--step"
            help = "Step between iter values in bench mode. If 0, uses iterations/4"
            arg_type = Int
            default = 0
        "--precond"
            help = "Preconditioned? [used in compute mode]."
            action = :store_true
        "--restarted"
            help = "Restarted? [used in compute mode]."
            action = :store_true
        "mode"
            help = "Program mode. Can be \"compute\" or \"bench\"."
            required = true
        "data"
            help = "Path to the problem data."
            required = true
    end

    return parse_args(s)
end

function getDEA(E_bar, edge_data, parsed_args)
    if parsed_args["diagonal"] == "ones"
        return gen_A_all_ones(E_bar)
    elseif parsed_args["diagonal"] == "uniform"
        return gen_A_uniform(E_bar)
    elseif parsed_args["diagonal"] == "quadratic"
        return gen_A_from_data(edge_data[:, 5], E_bar)
    else
        @assert false
    end
end

parsed_args = parse_commandline()


flows, E_bar, edge_data = load_mcfp_data(parsed_args["data"])

y = (gen_y_from_data(flows, edge_data[:, 5]))


"""
Based on command-line arguments, computes residuals or plots comparisons between the algorithms.
"""

if parsed_args["mode"] == "compute"
    println("Compute mode\n============")

    niters = parsed_args["iterations"]
    println("Performing $niters iterations:")

    if !parsed_args["restarted"] && !parsed_args["precond"]
        D, E, A = getDEA(E_bar, edge_data, parsed_args)

        # warmup iteration: ensures that the code is compiled before it is timed
        x = GMRES_IncrementalQR(A, y, 1)
        t = @elapsed begin
            x = GMRES_IncrementalQR(A, y, niters)
        end
        println("residual: ", norm(A * x - y), " computed in ", t, "s")


    elseif !parsed_args["restarted"] && parsed_args["precond"]
        D, E, A = getDEA(E_bar, edge_data, parsed_args)

        # warmup iteration: ensures that the code is compiled before it is timed
        x = GMRES_IncrementalQR_precond(D, SparseMatrixCSC(E), y, 1)
        t = @elapsed begin
            x = GMRES_IncrementalQR_precond(D, SparseMatrixCSC(E), y, niters)
        end
        println("residual: ", norm(A * x - y), " computed in ", t, "s")


    elseif parsed_args["restarted"] && !parsed_args["precond"]
        D, E, A = getDEA(E_bar, edge_data, parsed_args)

        # warmup iteration: ensures that the code is compiled before it is timed
        x = GMRES_Restarted(GMRES_IncrementalQR, A, y, (A,), 1, 1, parsed_args["tol"])
        t = @elapsed begin
            x = GMRES_Restarted(GMRES_IncrementalQR, A, y, (A,), niters, parsed_args["restart"], parsed_args["tol"])
        end
        println("residual: ", norm(A * x - y), " computed in ", t, "s")


    elseif parsed_args["restarted"] && parsed_args["precond"]
        D, E, A = getDEA(E_bar, edge_data, parsed_args)

        # warmup iteration: ensures that the code is compiled before it is timed
        x = GMRES_Restarted(GMRES_IncrementalQR_precond, A, y, (D, SparseMatrixCSC(E)), 1, parsed_args["restart"], parsed_args["tol"])

        t = @elapsed begin
            x = GMRES_Restarted(GMRES_IncrementalQR_precond, A, y, (D, SparseMatrixCSC(E)), niters, parsed_args["restart"], parsed_args["tol"])
        end
        println("residual: ", norm(A * x - y), " computed in ", t, "s")
    end

elseif parsed_args["mode"] == "bench"
    println("Bench mode\n==========")

    niters = parsed_args["iterations"] == 0 ? size(E_bar)[1] + size(E_bar)[2] - 1 : parsed_args["iterations"]
    stepval = parsed_args["step"] == 0 ? floor(Int, niters / 4) : parsed_args["step"]
    println("Performing $niters iterations with step = $stepval")

    D, E, A = getDEA(E_bar, edge_data, parsed_args)

    plot_residual_restarted(D, E, A, y, niters, stepval, parsed_args["tol"], parsed_args["restart"], parsed_args["restart_precond"])
    println("press enter to terminate program.")
    readline()
else
    println("Error: mode not recognized.")
end


