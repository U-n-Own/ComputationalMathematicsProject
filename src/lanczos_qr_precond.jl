const PRECOND = "PAPT"

using LinearAlgebra
using Random
using SparseArrays
using ExponentialUtilities  # to compare with builtin arnoldi
using TickTock  # to test complexity of LanczosQR

include("QR.jl")
include("lanczos_qr.jl")

# Random.seed!(42)

function LanczosQRPrecond_using_lanczosqr(D::Diagonal, E :: Matrix, y::Vector, n::Int)
    sizeA = size(D)[1] + size(E)[1]
    sizeD = size(D)[1]
    heightE = size(E)[1]
    S = E * inv(D) * E'

    # construct matrix and vector
    if PRECOND == "PAPT"
        PAPT = SparseMatrixCSC([D*D*D D*D*E' + D*E'*S' ; (E*D + S*E)*D (E*D + S*E) * E' + E*E'*S'])
        yu = y[1 : sizeD]
        yb = y[sizeD + 1 : sizeA]
        Py = SparseVector([D * yu ; E * yu + S * yb])
        return LanczosQR(PAPT, Py, n)
    else
        PTAP = SparseMatrixCSC([(D*D+E'*E)*D + D*E'*E D'*E'*S ; S'*E*D zeros(heightE,heightE)])
        yu = y[1 : sizeD]
        yb = y[sizeD + 1 : sizeA]
        PTy = SparseVector([D * yu + E' * yb ; S' * yb])
        return LanczosQR(PTAP, PTy, n)
    end


end

# TODO try to optimize this
function LanczosQRPrecond(D::Diagonal, E :: Matrix, y::Vector, n::Int)
    """
    Builds QR factorization of H while performing Lanczos
    """     
    
    alpha = zeros(1,n)     # diagonal of H
    beta = zeros(1,n)    # subdiagonal of H (= superdiagonal of H because symmetric)
    
    sizeA = size(D)[1] + size(E)[1]
    sizeD = size(D)[1]
    heightE = size(E)[1]

    L = zeros(sizeA, n)    # Lanczos Basis
    w = zeros(sizeA)
    
    Q = zeros(n+1, n+1)
    R = zeros(n+1, n)
    
    Eᵀ = E'
    S = E * inv(D) * E'     # inv should be efficient (linear) because D is diagonal

    yu = y[1 : sizeD]
    yb = y[sizeD + 1 : sizeA]

    if PRECOND == "PAPT"
        y = [D * yu ; E * yu + S * yb]      # y = P * y
    else
        y = [D * yu + E' * yb ; S' * yb]    # y = P' * y
    end

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

            qu = q[1 : sizeD]
            qb = q[sizeD + 1 : sizeA]

            # compute (PAP') q without constructing P

            if PRECOND == "PAPT"
                # P A P ' is [D*D*D D*D*E' + D*E'*S' ; (E*D + S*E)*D (E*D + S*E) * E' + E*E'*S']
                w[1:sizeD] = (D.^3) * qu + (D*D*Eᵀ + D*Eᵀ*S') * qb # TODO analyze complexity
                w[sizeD + 1 : sizeA] = ((E*D + S*E)*D) * qu + ((E*D + S*E) * E' + E*E'*S') * qb
            else
               # P' A P is [(D*D+E'*E)*D + D*E'*E D'*E'*S ; S'*E*D zeros(heightE,heightE)]
                w[1:sizeD] = ((D*D+E'*E)*D + D*E'*E) * qu + (D'*E'*S) * qb
                w[sizeD + 1 : sizeA] = (S'*E*D) * qu
            end
            alpha[1] = w' * q       # O(m)
            w = w - alpha[1] * q    # O(m)
            beta[1] = norm(w)       # O(m)
        else
            # inductive case
            q = w / beta[j-1]       # O(m)
            L[:, j] = q             # O(m)

            qu = q[1 : sizeD]
            qb = q[sizeD + 1 : sizeA]

            # compute (PAP') q without constructing P
            if PRECOND == "PAPT"
                # P A P ' is [D*D*D D*D*E' + D*E'*S' ; (E*D + S*E)*D (E*D + S*E) * E' + E*E'*S']
                w[1:sizeD] = (D.^3) * qu + (D*D*Eᵀ + D*Eᵀ*S') * qb # TODO analyze complexity
                w[sizeD + 1 : sizeA] = ((E*D + S*E)*D) * qu + ((E*D + S*E) * E' + E*E'*S') * qb
            else
               # P' A P is [(D*D+E'*E)*D + D*E'*E D'*E'*S ; S'*E*D zeros(heightE,heightE)]
                w[1:sizeD] = ((D*D+E'*E)*D + D*E'*E) * qu + (D'*E'*S) * qb
                w[sizeD + 1 : sizeA] = (S'*E*D) * qu
            end

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
            c = beta[j-1] * Q[j-1, 1:j]' + alpha[j] * Q[j, 1:j]' # O(j)   (transpose does not impact on complexity)
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
                return (L, alpha, beta, Q, R)
            end
            
        end
        end

        
    end
    return (L, alpha, beta, Q, R)

end

dim1 = 800
dim2 = 200
dim = dim1 + dim2

D = Diagonal(1.0 * I(dim1))
E = Matrix(sprand(dim2, dim1, .7))
Z = SparseMatrixCSC(zeros(dim2, dim2))

A = hcat(vcat(D, E), vcat(E', Z))

b = Vector(sprand(dim, .7))

L, alpha, beta, Q, R = LanczosQRPrecond(D, E, b, dim)


H = Q*R
# println(size(R))

sizeA = size(D)[1] + size(E)[1]
sizeD = size(D)[1]
S = E * inv(D) * E'


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

