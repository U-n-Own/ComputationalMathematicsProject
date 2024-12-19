using LinearAlgebra
using Random
using SparseArrays
using ExponentialUtilities  # to compare with builtin arnoldi

include("QR.jl")

# Random.seed!(42)


function LanczosQR(A::SparseMatrixCSC, y::SparseVector, n::Int)
    """
    Builds QR factorization of H while performing Lanczos
    """     
    
    alpha = zeros(1,n)     # diagonal of H
    beta = zeros(1,n)    # subdiagonal of H (= superdiagonal of H because symmetric)
    
    L = zeros(size(A)[1], n)    # Lanczos Basis
    w = zeros(size(A)[1], 1)
    
    Q = SparseMatrixCSC(zeros(n+1, n+1))
    R = SparseMatrixCSC(zeros(n+1, n))
    
    for j in 1:n
        
        if j % 500 == 0
            println(j, "-th iteration")
        end
    
        if j == 1
            # base case
            q = y / norm(y)
            L[:, 1] = q
            w = A * q
            alpha[1] = w' * q
            w = w - alpha[1] * q
            beta[1] = norm(w)
        else
            # inductive case
            q = w / beta[j-1]
            L[:, j] = q
            
            w = A * q
            alpha[j] = w' * q
            w = w - alpha[j] * q - beta[j-1] * L[:, j-1]

            beta[j] = norm(w)
        end
            
        # QR factorization
        if j == 1
            H1 = reshape([alpha[1] ; beta[1]], 2,1)
            Q[1:2, 1:2], R[1:2, 1] = QR_fact(H1)
#             println("QR")
#             display(Q)
#             display(R)
        else
            # last column of R:
            c = beta[j-1] * Q[j-1, 1:j]' + alpha[j] * Q[j, 1:j]'
            x = [c[j]; beta[j]]
            y = [norm(x), 0]
            v = x - y
            # Hrefl is symmetric [((a - norm(x))^2 , (a - norm(x)beta_n); (a - norm(x)beta_n) , beta_n^2 )]
            Hrefl = Matrix(1.0 * I(2)) - (v * v' * 2 / (v' * v))
            R[1:j, j] = [c[1:j-1] ; norm(x)]
            
            
#              O = vcat(
#                 hcat(1.0 * I(j-1) , zeros(j-1, 2)),
#                 hcat(zeros(2, j-1) , Hrefl))
            
            

            Q[j+1, j+1] = 1.0
           
#             Q[1:j+1, 1:j+1] = (O * (Q[1:j+1, 1:j+1])')'
#             Q[1:j+1, 1:j+1] = Q[1:j+1, 1:j+1] * O
            
            newcols = zeros(j+1, 2)
            
            for i in 1:j+1
                # Compute Q_old * O without constructing O
                # Can compute Q_old * O instead of Q_old * O' because O is sym.
                
                newcols[i, 1]     = Q[i, j] * Hrefl[1,1] + Q[i, j+1] * Hrefl[2,1]
                newcols[i, 2]   = Q[i, j] * Hrefl[1,2] + Q[i, j+1] * Hrefl[2,2]
            end
            
            Q[1:j+1, j:j+1] = newcols
            
            
            if beta[j] < 1e-13
                println("Breakdown");
                return (L, alpha, beta, Q, R)
            end
            
        end
        
        
    end
    return (L, alpha, beta, Q, R)

end

