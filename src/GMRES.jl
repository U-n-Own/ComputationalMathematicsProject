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
include("lanczos_qr_precond_new.jl")


# Random.seed!(42)


function GMRES_reference(A::SparseMatrixCSC, y::Vector, n::Int)
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


function GMRES_lanczosQR_reference(A::SparseMatrixCSC, y::Vector, n::Int)
    L, α, β, Q, R = LanczosQR(A, y, n)
    
    H = Q * R
    e1 = Matrix(1.0 * I(size(H)[1]))[:, 1]
    println("time to compute the result")
    
    return L * (H \ (e1 * norm(y)))
end

function GMRES_IncrementalQR(A::SparseMatrixCSC, y::Vector, iterations::Int)
    
    """
    Uses LanczosQR to compute the QR factorization of H efficiently, then solves the minimum problem using LeastSquares_QR
    """
    L, α, β, Q, R = LanczosQR(A, y, iterations)
    
    e1 = (1.0*I(iterations))[:, 1]
    
    z = LeastSquares_QR(Q, R, (e1*norm(y)))
    
    x = L[:, 1:iterations] * z
    
    return x

end

function GMRES_IncrementalQR_precond(D::Diagonal, E::SparseMatrixCSC, y::Vector, iterations::Int)

    """
    Uses LanczosQR to compute the QR factorization of H efficiently, then solves the minimum problem using LeastSquares_QR
    """

    sizeD = size(D)[1]
    sizeA = size(D)[1] + size(E)[1]
    heightE = size(E)[1]
    S = E * inv(D) * E'

    # set off-diagonal elements of S under some threshold to zero
    if THRESHOLD_ZERO > 0
        threshold = THRESHOLD_ZERO
        mask = .!I(heightE) .& (abs.(S) .< threshold)
        S[mask] .= 0
    end

    L, α, β, Q, R = LanczosQRPrecond(D, E, y, iterations)

    actual_iterations = size(R)[2]

    e1 = Matrix(1.0*I(actual_iterations))[:, 1]

    U = (cholesky(Matrix(S)).U)       # U'U = S
    Uinv = SparseMatrixCSC(inv(U))
    sqrtD = inv(sqrt.(D))

    # Multiplying P by y
    bu = y[1 : sizeD]
    bb = y[sizeD + 1 : sizeA]
    y_prec = [sqrtD * bu ; Uinv' * bb]

    z = LeastSquares_QR(Q, R, (e1*norm(y_prec)))

    out_x = L[:, 1:actual_iterations] * z

    # Multiplying P' by x

    xu = out_x[1 : sizeD]
    xb = out_x[sizeD + 1 : sizeA]
    x = [sqrtD * xu ; Uinv * xb]

    return x

end

function GMRES_Restarted(GMRES_Method :: Function, A::SparseMatrixCSC, y::Vector, args :: Tuple, max_iter::Int, restart::Int, tol::Float64)
    """
    Implements Restarted GMRES.

    max_iter: Int - maximum number of iterations allowed
    restart: Int - number of iterations between restarts
    tol: Float64 - convergence tolerance for current residual

    Returns the solution vector x.
    """
    # Our initial guess is the zero vector, we progressively refine it with reusing solution
    # from previous iterations
    x = spzeros(size(A, 2))

    # Current residual
    r = y - A * x
    norm_r = norm(r)

    iter = 0
    while norm_r > tol && iter < max_iter

        # Perform GMRES for 'restart' iterations
        dx = GMRES_Method(args..., Vector(r), restart)

        # Update solution and residual
        x += dx
        r = y - A * x
        norm_r = norm(r)

        iter += restart

        println("Iteration $iter, Residual Norm: $norm_r")
    end

    if norm_r <= tol
        println("Converged in $iter iterations with residual norm $norm_r")
    else
        println("Did not converge within $max_iter iterations. Final residual norm: $norm_r")
    end

    return x
end




#=
D = SparseMatrixCSC(1.0 * I(1000))
E = sprand(200, 1000, .7)
Z = SparseMatrixCSC(zeros(200, 200))

A = hcat(vcat(D, E), vcat(E', Z))

b = sprand(1200, .7)

x = GMRES_lanczosQR_reference(A, b, 1200)

println(norm(A * x - b))

=#
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
