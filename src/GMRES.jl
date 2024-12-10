using LinearAlgebra 
using Random
using SparseArrays
using Printf

include("Arnoldi.jl")
include("QR.jl")
include("lanczos_qr.jl")


Random.seed!(42)


function GMRES(A::SparseMatrixCSC, y::Vector, n::Int)
    
    Q, H = Arnoldi(A, y, n)

    v = vcat(I(1), zeros(n))[:] * norm(y)

    # Solving here the following: min_z ||H_n*z - e1*norm(y)||
    z_our = LeastSquares_QR(SparseMatrixCSC(H[:, 1:n]), SparseVector(v))
    
    # same here with standard \
    #z = H[:, 1:n] \ v
    
    x = Q[:, 1:n] * z_our

    return x

end


function GMRES_IncrementalQR(A::SparseMatrixCSC, y::SparseVector, iterations::Int)
    
    """
    This version of GMRES we transform the problem in this way:
    
    min y = ||H_ny - be_1||_2 -> min y = ||Ry - Q^T(be1)||_2
    Solving then the Triangular system Ry = Q^T(be1), we can do this since Q is orthogonal 
    multiplying do not affect the norm 2.
    
    - After we ran Arnoldi and get Q and Hn 
    - Extend H_n+1 by adding a new column and row
    - Use Given rotation to modify the extended H_n+1 to stay upper Hessenberg
    - Update R and Q

    1. QR fact from Lanczos QR
    2. Solve via backsubstitution
    """

    m = size(A, 1)

    L, α, β, Q, R = LanczosQR(A, y, iterations)
    e1 = (1.0*I(iterations+1))[:, 1]
    z = LeastSquares_QR(Q, R, SparseVector(e1*norm(y)))
    
    x = L[:, 1:iterations] * z
    
    return x

end


#= A = sprand(100, 100, 0.5)

println("Det sparse",det(A))

println(cond(Matrix(A), 2))

y = rand(100)
n = 200

@time x = GMRES(A, y, n)

println(norm(A*x - y))
  =#


mat_size = 5
diag_size = 2
D = SparseMatrixCSC(1.0 * I(diag_size))
E = sprand(mat_size - diag_size, diag_size, .7)

Asym = hcat(vcat(D, E), vcat(E', SparseMatrixCSC(zeros(mat_size - diag_size, mat_size - diag_size))))
println(rank(Asym))
display(Asym)


b = rand(mat_size)

x = GMRES_IncrementalQR(Asym, SparseVector(b), mat_size)

# display(x)

println("Error is :", norm(Asym*x - b))


#=
A = sprand(100, 100, 0.9)
b = rand(100)
# random symmetric matrix
Asym = A*A'

x = GMRES_IncrementalQR(Asym, SparseVector(b), s)

display(x)

println("Error is :", norm(Asym*x - b))
=#
