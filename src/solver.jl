using LinearAlgebra
using IterativeSolvers
using SparseArrays
using Random
# Utilities
using NPZ
using Plots
using LightGraphs

# We will use GMRES and could use SparseArrays and construct the KKT system
# The prblem has a matrix is in this form: Augmented System

# [ D  E' ] [ x ] = [ b ]
# [ E  0  ] [ y ] = [ c ]

function householder_v(v::Vector)
    """
    We return directly e1 since it's the first element of the basis multiplied by norm(v)
    """
    e1 = zeros(size(v))
    e1[1] = norm(v)
    
    return e1
    
end

function householder_m(v::Vector)

    return 1.0I(size(v)[1])-2*(v*v')/(norm(v)^2)
    
end

function QR_householder(A::Matrix)
    """
    Compute the QR factorization of a matrix A using Householder reflectors:
    - This method is numerically stable 
    
    TODO:This version has no matrix in SparseArray form maybe we should write.

    Returns:
    Q: Orthogonal matrix
    R: Upper triangular matrix
    """ 

    m, n = size(A)
    Q = 1.0*I(m)
    R = copy(A)
    
    for k in 1:n
    # Construct H
    u = householder_v(R[k:end, k])
    H = 1.0*I(size(u)[1]) - (2*u*u')
    #H = householder_m(R[k:end, k])

    println("H matrix")
    display(H)

    # Submatrix
    R[k:end, k:end] = H * R[k:end, k:end]
    # accumulate Q
    Q[:, k:end] = Q[:, k:end]*H
    
    println("Q matrix")
    display(Q)
    # Get R iterate, accumulate Q:= Q1*Q2...*Qn
    # Remeber Q:=Q1..Qn*R=A
    # Also Q can be constructed [Q0 | Qc][R0 ; 0] yielding ThinQR
    end

    return Q,R
end

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

n = 5

P = rand(n,n)

Q,R = QR_householder(P)

