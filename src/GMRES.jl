using LinearAlgebra 
using Random
using SparseArrays
using Printf


include("Lanczos.jl")
include("QR.jl")
include("lanczos_qr.jl")
include("lanczos_qr_precond_new.jl")


# Random.seed!(42)

function GMRES_IncrementalQR(A::SparseMatrixCSC, y::Vector, iterations::Int)
    
    """
    - Uses LanczosQR to compute the QR factorization of H efficiently,
    - Then solves the minimum problem using LeastSquares_QR
    """
    L, α, β, Q, R = LanczosQR(A, y, iterations)
    
    e1 = (1.0*I(iterations))[:, 1]
    
    z = LeastSquares_QR(Q, R, (e1*norm(y)))
    
    x = L[:, 1:iterations] * z
    
    return x

end

function GMRES_IncrementalQR_precond(D::Diagonal, E::SparseMatrixCSC, y::Vector, iterations::Int)

    """
    - Uses LanczosQRPrecond to compute QR,
    - Solves the preconditioned problem using LeastSquares_QR,
    - Transforms the solution into the original solution
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

    GMRES_Method: Function - the GMRES method to be used
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
    end

    if norm_r <= tol
        println(" * Converged in $iter iterations with residual norm $norm_r")
    else
        println(" * Did not converge within $max_iter iterations. Final residual norm: $norm_r")
    end

    return x
end
