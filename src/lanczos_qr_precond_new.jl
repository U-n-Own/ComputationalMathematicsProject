using LinearAlgebra
using Random
using SparseArrays
using ExponentialUtilities  # to compare with builtin arnoldi
using TickTock  # to test complexity of LanczosQR

include("QR.jl")
include("lanczos_qr.jl")

# Random.seed!(42)

"""
Select what form to use for preconditioned matrix.

"""

const PRECOND = "PAPT"

# This threshold is used to eliminate near-zero off-diag elements from the matrix S
# currently not used
const THRESHOLD_ZERO = 0

# P = [sqrt D  0 ; 0 R^-T]

# pass Uinv instead of UinvT to use this to multiply by P'
function mult_by_P(v, sqrtD, Uinv)
    sizeD = size(sqrtD)[1]
    sizeP = sizeD + size(Uinv)[1]

    bu = v[1 : sizeD]
    bb = v[sizeD + 1 : sizeP]
    v = [sqrtD * bu ; Uinv' * bb]

    return v
end

function mult_by_PT(v, sqrtD, Uinv)
    return mult_by_P(v, sqrtD, Uinv')
end


# JUST FOR TESTING
function LanczosQRPrecond_using_lanczosqr(D::Diagonal, E :: SparseMatrixCSC, y::Vector, n::Int)
    sizeA = size(D)[1] + size(E)[1]
    sizeD = size(D)[1]
    heightE = size(E)[1]
    S = E * inv(D) * E'
    U = (cholesky(Matrix(S)).U)       # U'U = S
    Uinv = SparseMatrixCSC(inv(U))
    sqrtD = inv(sqrt(D))

    # set off-diagonal elements under some threshold to zero
    if THRESHOLD_ZERO > 0
        threshold = THRESHOLD_ZERO
        mask = .!I(heightE) .& (abs.(S) .< threshold)
        S[mask] .= 0
    end

    # construct matrix and vector
    P = [sqrtD zeros(sizeD, heightE) ; zeros(heightE, sizeD) Uinv']
    A = [D E' ;
         E zeros(size(E, 1), size(E, 1))]

    PAPT = P * A * P'

    Py = P * y
    return LanczosQR(PAPT, Py, n)
end


function LanczosQRPrecond(D::Diagonal, E :: SparseMatrixCSC, y::Vector, n::Int)
    """
    Builds QR factorization of H while performing Lanczos
    """     
    
    alpha = zeros(1,n)      # diagonal of H
    beta = zeros(1,n)       # subdiagonal of H (= superdiagonal of H because symmetric)
    
    sizeA = size(D)[1] + size(E)[1]
    sizeD = size(D)[1]
    heightE = size(E)[1]

    L = zeros(sizeA, n)     # Lanczos Basis
    w = zeros(sizeA)
    
    Q = zeros(n+1, n+1)
    R = zeros(n+1, n)
    
    S = E * inv(D) * E'     # inv should be efficient (linear) because D is diagonal

     # set off-diagonal elements under some threshold to zero
    if THRESHOLD_ZERO > 0
        threshold = THRESHOLD_ZERO
        mask = .!I(heightE) .& (abs.(S) .< threshold)
        S[mask] .= 0
    end

    U = (cholesky(Matrix(S)).U)       # U'U = S
    Uinv = SparseMatrixCSC(inv(U))
    sqrtD = inv(sqrt.(D))

    y = mult_by_P(y, sqrtD, Uinv)

    # base case + n * (inductive case)
    # let A be m*m
    for j in 1:n
        if j % 500 == 0
            if j > 500
                tock()
            end
            println(j, "-th iteration")
            tick()
        end


        if j == 1
            # base case
            q = y / norm(y)
            L[:, 1] = q             # O(m)

            w = mult_by_PT(q, sqrtD, Uinv)
            w = A * w
            w = mult_by_P(w, sqrtD, Uinv)

            alpha[1] = w' * q       # O(m)
            w = w - alpha[1] * q    # O(m)
            beta[1] = norm(w)       # O(m)
        else
            # inductive case
            q = w / beta[j-1]       # O(m)
            L[:, j] = q             # O(m)

            w = mult_by_PT(q, sqrtD, Uinv)
            w = A * w
            w = mult_by_P(w, sqrtD, Uinv)

            alpha[j] = w' * q       # O(m)
            w = w - alpha[j] * q - beta[j-1] * L[:, j-1]        # O(m)

            beta[j] = norm(w)       # O(m)
        end

        begin

        # QR factorization
        if j == 1
            H1 = reshape([alpha[1] ; beta[1]], 2,1)     # O(1)
            Q[1:2, 1:2], R[1:2, 1] = QR_fact(H1)        # O(1)

        else
            # last column of R:
            c = beta[j-1] * Q[j-1, 1:j]' + alpha[j] * Q[j, 1:j]' # O(j)   (transpose does not impact on complexity(???))
            x = [c[j]; beta[j]]     # O(1)
            y = [norm(x), 0]        # O(1)
            v = x - y               # O(1)
            # Hrefl is symmetric [((a - norm(x))^2 , (a - norm(x)beta_n); (a - norm(x)beta_n) , beta_n^2 )]
            Hrefl = Matrix(1.0 * I(2)) - (v * v' * 2 / (v' * v))    # O(1)
            R[1:j, j] = [c[1:j-1] ; norm(x)]    #(O(j))

            Q[j+1, j+1] = 1.0
            newcols = zeros(j+1, 2) # O(j) Keeping this here is better for performance somehow

            # O(j)
            for i in 1:j+1
                # Compute Q_old * O without constructing O
                # Can compute Q_old * O instead of Q_old * O' because O is sym.

                newcols[i, 1]     = Q[i, j] * Hrefl[1,1] + Q[i, j+1] * Hrefl[2,1]  # O(1)
                newcols[i, 2]     = Q[i, j] * Hrefl[1,2] + Q[i, j+1] * Hrefl[2,2]  # O(1)
            end

            @views Q[1:j+1, j:j+1] .= newcols[1:j+1, :]   #O(j)     without using .= and views this took way too long


            if beta[j] < 1e-13
                println("Breakdown");
                return (L[:, 1:j], alpha, beta, Q[1:j+1, 1:j+1], R[1:j+1, 1:j])
            end

        end
        end


    end
    return (L, alpha, beta, Q, R)

end

#=
dim1 = 8000
dim2 = 200
dim = dim1 + dim2

D = Diagonal(1.0 * I(dim1))
E = Matrix(sprand(dim2, dim1, .002))
Z = SparseMatrixCSC(zeros(dim2, dim2))

A = hcat(vcat(D, E), vcat(E', Z))

b = Vector(sprand(dim, .7))

L, alpha, beta, Q, R = LanczosQRPrecond(D, E, b, dim)


H = Q*R
# println(size(R))

sizeA = size(D)[1] + size(E)[1]
sizeD = size(D)[1]
S = E * inv(D) * E'
# set off-diagonal elements under some threshold to zero
threshold = THRESHOLD_ZERO
mask = .!I(dim2) .& (abs.(S) .< threshold)
S[mask] .= 0


e1 = Matrix(1.0 * I(dim + 1))[:, 1]

if (PRECOND == "PAPT")
    bu = b[1 : sizeD]
    bb = b[sizeD + 1 : sizeA]
    Pb = [D * bu ; E * bu + S * bb]
    invPTx = L * (H \ (e1 * norm(Pb)))
    xu = invPTx[1 : sizeD]
    xb = invPTx[sizeD + 1 : sizeA]
    x = [D * xu + E' * xb ; S' * xb]
else
    bu = b[1 : sizeD]
    bb = b[sizeD + 1 : sizeA]
    PTb = [D * bu + E' * bb ; S' * bb]
    invPx  = L * (H \ (e1 * norm(PTb)))
    xu = invPx[1 : sizeD]
    xb = invPx[sizeD + 1 : sizeA]
    x = [D * xu; E * xu + S * xb]
end

println(norm(A * x - b))

=#
