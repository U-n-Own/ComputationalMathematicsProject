using LinearAlgebra
using IterativeSolvers
using SparseArrays
using Random
# Utilities
using NPZ
using Plots
using LightGraphs

# We will use GMRES and could use SparseArrays and construct the KKT system
# The prblem has a matrix is in this form:

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

    D = data["diagonal_matrix"]
    E = data["node_arc_matrix"]
    edge_data = data["full_edges"]

    return D, E, edge_data
end

# Visualize the D matrix and E matrix with eye
D, E, edge_data = load_mcfp_data("dataset/net10_8_5.dmx_out.npz")


# Sanity checks...

# check sparsity of E
println("Sparsity of E: ", count(!iszero, E) / length(E))

# get E sparse
Esparse = sparse(E)

# visualize Esparse
heatmap(Esparse, aspect_ratio=1, color=:grays, c=:blues, yflip=true, title="E matrix")

# check space occupied of E and Esparse in megabytes
println("Space occupied by E: ", sizeof(E) / 1024^2, " MB")
println("Space occupied by Esparse: ", sizeof(Esparse) / 1024^2, " MB")