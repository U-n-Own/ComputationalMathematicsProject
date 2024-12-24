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

    display(size(E))
    display(size(flows))
    display(size(edge_data[:, 5]))

    return flows, E, edge_data
end

# --------------- Load a problem from netgen (nb. netgen generate random easy graphs without structure)
flows, E, edge_data = load_mcfp_data("dataset/net10_8_1.dmx_out.npz")


# --------------- Construct the augmented system:

# A = gen_A_from_data(edge_data[:, 5], E)
# A = gen_A_all_ones(E)
A = gen_A_uniform(E)

println(size(A))
println("matrix has rank: ", rank(A))

# println("condition number of A ", cond(Matrix(A)))

display(A)

# y = SparseVector(rand(size(A, 2)))
y = SparseVector(gen_y_from_data(flows, edge_data[:, 5]))

n = size(A)[1]

@time x = GMRES_IncrementalQR(A, y, n)
#@time x1 = GMRES_reference(A, y, n)

# println("size is: ", size(x))

println(norm(A * x - y))
#println(norm(A*x1 - y))
