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

# We will use GMRES and could use SparseArrays and construct the KKT system
# The prblem has a matrix is in this form: Augmented System

# [ D  E' ] [ x ] = [ b ]
# [ E  0  ] [ y ] = [ c ]


# Load from npz file the in order D diagonal matrix, E node arc incidence matrix and full edge informations
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

# --------------- Load a problem from netgen (nb. netgen generate random easy graphs without structure)
flows, E, edge_data = load_mcfp_data("dataset/net10_8_1.dmx_out.npz")


# --------------- Construct the augmented system:

# A = gen_A_from_data(edge_data[:, 5], E)
A = gen_A_all_ones(E)
# A = gen_A_uniform(E)

P = gen_test_prec_all_ones(E)
# P = gen_test_prec_from_data(edge_data[:, 5], E)

PAPT = P * A * P'
#=
println("cond of matrix:", norm(A) * norm(inv(Matrix(A))))
println("cond of precond matrix:", norm(PAPT) * norm(inv(Matrix(PAPT))))=#

println("matrix A has rank: ", rank(A))

display(A)

# y = SparseVector(rand(size(A, 2)))
y = SparseVector(gen_y_from_data(flows, edge_data[:, 5]))

n = size(A)[1]

println("now solving non-precond problem:")

#@time x = GMRES_IncrementalQR(A, y, n)

@time x1 = GMRES_Restarted(A, y, n, 100, 1e-6)
#@time x1 = GMRES_reference(A, y, n)

# println("size is: ", size(x))
#plot times and residuals

#println("Non restarted residual", norm(A * x - y))
println("Restarted residual: ", norm(A*x1 - y))
