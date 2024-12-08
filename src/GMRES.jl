using LinearAlgebra 
using Random
using SparseArrays
using Printf

include("Arnoldi.jl")
include("QR.jl")


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


function GMRES_IncrementalQR(A::SparseMatrixCSC, y::Vector, n::Int)
    
    """
    This version of GMRES we transform the problem in this way:
    
    min y = ||H_ny - be_1||_2 -> min y = ||Ry - Q^T(be1)||_2
    Solving then the Triangular system Ry = Q^T(be1), we can do this since Q is orthogonal 
    multiplying do not affect the norm 2.
    
    - After we ran Arnoldi and get Q and Hn 
    - Extend H_n+1 by adding a new column and row
    - Use Given rotation to modify the extended H_n+1 to stay upper Hessenberg
    - Update R and Q

    1. Arnoldi 1 step
    2. QR fact of H from Arnoldi
    3.
    """

    m = size(A, 1)
    Q = zeros(m, n + 1) # Orthonormal basis
    H = zeros(n + 1, n) # Hessenberg matrix
    R = zeros(n + 1, n) # Upper triangular matrix for QR factorization

    


end


#= A = sprand(100, 100, 0.5)

println("Det sparse",det(A))

println(cond(Matrix(A), 2))

y = rand(100)
n = 200

@time x = GMRES(A, y, n)

println(norm(A*x - y))
  =#

A = sprand(5, 5, 0.3)
b = rand(5)
# random symmetric matrix
Asym = A*A'

for i in 1:5
  # run Arnoldi
  Q, H = Arnoldi(Asym, b, i)

  @printf "H %d\n" i
  display(H)

  # H_n = Qt*Rt 
#   Q, R = QR_fact(SparseMatrixCSC(H))

#   @printf "Q %d\n" i
#   display(Q)
#   @printf "R %d\n" i
#   display(R)
end





