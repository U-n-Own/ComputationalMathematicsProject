using LinearAlgebra
using Random
using SparseArrays
using ExponentialUtilities  # to compare with builtin arnoldi

include("QR.jl")

Random.seed!(42)


function LanczosQR(A::SparseMatrixCSC, y::Vector, n::Int)
    
    alpha = zeros(1,n)     # diagonal of H
    beta = zeros(1,n)    # subdiagonal of H (= superdiagonal of H because symmetric)
    
    L = zeros(size(A)[1], n)    # Lanczos Basis
    w = zeros(size(A)[1], 1)
    
    Q = zeros(n+1, n+1)
    R = zeros(n+1, n)
    
    for j in 1:n
    
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
            
#         if beta[j] < 1e-13
#             println("Breakdown");
#             return 
#         end
        
        # QR factorization
        if j == 1
            H1 = reshape([alpha[1] ; beta[1]], 2,1)
            Q[1:2, 1:2], R[1:2, 1] = QR_fact(H1)
            println("QR")
            display(Q)
            display(R)
        else
            # last column of R:
            c = beta[j-1] * Q[j-1, 1:j]' + alpha[j] * Q[j, 1:j]'
            x = [c[j]; beta[j]]
            y = [norm(x), 0]
            v = x - y
            # Hrefl is symmetric [((a - norm(x))^2 , (a - norm(x)beta_n); (a - norm(x)beta_n) , beta_n^2 )]
            Hrefl = Matrix(1.0 * I(2)) - (v * v' * 2 / (v' * v))
            R[1:j, j] = [c[1:j-1] ; norm(x)]
            
            # make this less space-shit -- can't be bothered rn
            # should be able to compute these explicitly
            
            O = vcat(
                hcat(1.0 * I(j-1) , zeros(j-1, 2)),
                hcat(zeros(2, j-1) , Hrefl))
           
            

            Q[j+1, j+1] = 1.0
           
            #Q[1:j+1, 1:j+1] = (O * (Q[1:j+1, 1:j+1])')'
            
            for i in 1:j+1
                # One for the second last column and one for the last column of Q'old * O  
                Q[j, i] = Q[j, i]*Hrefl[1,1] + Q[j+1, i]*Hrefl[2,1] 
                Q[j+1, i] = Q[j, i]*Hrefl[1,2] + Q[j+1, i]*Hrefl[2,2]     
            end
            
            print("error on approx Q is : ", norm((O * (Q[1:j+1, 1:j+1]))) - norm(Q[1:j+1, 1:j+1]))

            println("==========")
            println("Q*R")
            display(Q*R)
            println("alpha, beta")
            display(alpha)
            display(beta)
            
        end
        
        
    end
    return (L, alpha, beta, Q, R)

end

A = sprand(7,7, .1)
A = A * A'
y = rand(7)
n = 5
println("Lanczos n = 5")
L, alpha, beta, Q, R = LanczosQR(A, y, n)
display(L)
display(alpha)
display(beta)


# println("builtin arnoldi")
# Ks = arnoldi(A, y)
# display(Ks)
 
