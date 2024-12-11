using LinearAlgebra 
using Random
using SparseArrays
using Printf

# import gmres
using IterativeSolvers
# import arnoldi
using ExponentialUtilities


# include("Arnoldi.jl")
include("QR.jl")
include("lanczos_qr.jl")


# Random.seed!(42)


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

    # ------------------- Builin QR -------------------
    Ks = arnoldi(A, y)
    H_arn = Ks.H[1:(size(Ks.H)[1] - 1), :]

    # Solve min H_n - e1*norm(y) 
    e1 = (1.0*I(iterations))[:, 1]
    zbi = H_arn \ (e1*norm(y))
    # get V is Q
    Qbi = Ks.V[:, 1:iterations]

    xbi = Qbi * zbi
  #=   # Do the QR factorization using builtin
    Qbi, Rbi = qr(H_arn)
    #convert Q and R to SparseMatrixCSC
    Qbi, Rbi = SparseMatrixCSC(Qbi), SparseMatrixCSC(Rbi)

    Qb = Qbi'*y
    xbi = zeros(size(A, 1))

    xbi = Rbi \ Qb =#
    # ------------------- Our QR -------------------
    m = size(A, 1)

    #x_bi = gmres(A, y)
    L, α, β, Q, R = LanczosQR(A, y, iterations)
    
    println("Q")
    display(Q)
    println("R")
    display(R)
    
    #check for those Qbi and Rbi are the same as in our QR
    
#=     println("------------------")
    println("Qbi")
    display(Qbi)
    println("Q")
    display(Q)
    println("------------------")
    println("Rbi")
    display(Rbi)
    println("R")
    display(R)
    println("------------------") =#
    

    e1 = (1.0*I(iterations))[:, 1]
    
    z = LeastSquares_QR(Q, R, SparseVector(e1*norm(y)))
    
    x = L[:, 1:iterations] * z
    
    return x, xbi

end


#= A = sprand(100, 100, 0.5)

println("Det sparse",det(A))

println(cond(Matrix(A), 2))

y = rand(100)
n = 200

@time x = GMRES(A, y, n)

println(norm(A*x - y))
  =#


mat_size = 6
diag_size = 3
D = SparseMatrixCSC(1.0 * I(diag_size))
E = sprand(mat_size - diag_size, diag_size, .7)

Asym = hcat(vcat(D, E), vcat(E', SparseMatrixCSC(zeros(mat_size - diag_size, mat_size - diag_size))))

while rank(Asym) != mat_size
  global E = sprand(mat_size - diag_size, diag_size, .7)
  global Asym = hcat(vcat(D, E), vcat(E', SparseMatrixCSC(zeros(mat_size - diag_size, mat_size - diag_size))))
  println(rank(Asym))
end

println("A")
display(Asym)

b = rand(mat_size)

@time x, x_bi = GMRES_IncrementalQR(Asym, SparseVector(b), mat_size)

# display(x)

println("Error is :", norm(Asym*x - b))

println("Error with builtin is :", norm(Asym*x_bi - b))
#=
A = sprand(100, 100, 0.9)
b = rand(100)
# random symmetric matrix
Asym = A*A'

x = GMRES_IncrementalQR(Asym, SparseVector(b), s)

display(x)

println("Error is :", norm(Asym*x - b))
=#
