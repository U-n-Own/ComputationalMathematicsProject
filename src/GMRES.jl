using LinearAlgebra 
using Random
using SparseArrays
using Printf

# import gmres
# using IterativeSolvers
# import arnoldi
using ExponentialUtilities


include("Lanczos.jl")
include("QR.jl")
include("lanczos_qr.jl")


# Random.seed!(42)


function GMRES_reference(A::SparseMatrixCSC, y::SparseVector, n::Int)
    L, alpha, beta = Lanczos(A, y, n)

    H = Matrix(SymTridiagonal(alpha[1, :], beta[1, :]))
    H = vcat(H, zeros(size(alpha[1, :]))')
    H[size(H)[1], size(H)[2]] = beta[size(beta)[2]]
    
    actual_iterations = size(H)[1]
    
    # Solve min H_n - e1*norm(y) 
    e1 = (1.0*I(actual_iterations))[:, 1]

    z = H \ (e1*norm(y))
    Q = L

    x = Q * z
  
    return x
end


function GMRES_IncrementalQR(A::SparseMatrixCSC, y::SparseVector, iterations::Int)
    
    """
    Uses LanczosQR to compute the QR factorization of H efficiently, then solves the minimum problem using LeastSquares_QR
    """

    m = size(A, 1)

    L, α, β, Q, R = LanczosQR(A, y, iterations)
    
    e1 = (1.0*I(iterations))[:, 1]
    
    z = LeastSquares_QR(Q, R, SparseVector(e1*norm(y)))
    
    x = L[:, 1:iterations] * z
    
    return x

end


#=
mat_size = 1000
diag_size = 800
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

@time x = GMRES_IncrementalQR(Asym, SparseVector(b), mat_size)

println("Error is :", norm(Asym*x - b))

@time x = GMRES_reference(Asym, SparseVector(b), mat_size)
println("Reference GMRES error is :", norm(Asym*x - b))

@time x = Asym \ b
println("Default solver error is :", norm(Asym*x - b))

=#
