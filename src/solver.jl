using LinearAlgebra
using IterativeSolvers
using SparseArrays
using Random
# Utilities
using NPZ
using Plots
using LightGraphs

include("GMRES.jl")

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
    edge_data: Full edge informations with a tuple of (source, target, capacity, cost)
    """

    data = NPZ.npzread(filename)

    flows = data["flows"]
    E = data["node_arc_matrix"]
    edge_data = data["full_edges"]

    return flows, E, edge_data
end

# --------------- Load a problem from netgen (nb. netgen generate random easy graphs without structure)
flows, E, edge_data = load_mcfp_data("dataset/net10_8_1.dmx_out.npz")


# --------------- Construct the augmented system:
Deye = Diagonal(ones(Int, size(E, 2)))

E = Int.(E)

Eᵀ = transpose(E)

Z = Int.(zeros(size(E, 1), size(E, 1))) 

A = [Deye Eᵀ ;
     E Z]
#convert A in SparseArrays
println(size(A))
#cond(A)

println(det(E*Eᵀ))

A = SparseMatrixCSC(A)

display(A)

y = SparseVector(rand(size(A, 2)))

n = size(A)[1]

#@time x = GMRES_IncrementalQR(A, y, n)
@time x1 = GMRES_reference(A, y, n)

# println("size is: ", size(x))

#println(norm(A*x - y))
println(norm(A*x1 - y))
