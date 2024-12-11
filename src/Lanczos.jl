using LinearAlgebra
using Random
using SparseArrays
using ExponentialUtilities

# Random.seed!(42)


# Correct! checked result with builtin Arnoldi.
function Lanczos(A::SparseMatrixCSC, y::SparseVector, n::Int)
    
    alpha = zeros(1,n)      # diagonal of H
    beta = zeros(1,n)       # subdiagonal of H (= superdiagonal of H because symmetric)
    
    L = zeros(size(A)[1], n)    # Lanczos Basis
    
    # base case
    
    q = y / norm(y)
    L[:, 1] = q
    w = A * q
    alpha[1] = w' * q
    w = w - alpha[1] * q
    beta[1] = norm(w)
    # inductive case
    
    for j in 2:n
        if beta[j-1] < 1e-13
            println("Breakdown");
            return
        end
        q = w / beta[j-1]
        L[:, j] = q
        
        w = A * q
        alpha[j] = w' * q
        w = w - alpha[j] * q - beta[j-1] * L[:, j-1]
        beta[j] = norm(w)
        
    end 
    
    return (L, alpha, beta)

end

# A = sprand(7,7, .1)
# A = A * A'
# y = rand(7)
# n = 3
# println("Lanczos n = 3")
# L, alpha, beta = Lanczos(A, y, n)
# display(L)
# display(alpha)
# display(beta)
# 
# println("single step")
# 
# q_old = L[:, n]
# w = A * q_old - alpha[n] * q_old - beta[n-1] * L[:, n-1]
# q = w / beta[n]
# w = A * q
# alpha_new = w' * q
# w = w - alpha_new * q - beta[n] * L[:, n]
# beta_new = norm(w)
# 
# display(alpha_new)
# display(beta_new)
# 
# n = 4
# println("Lanczos n = 4")
# L, alpha, beta = Lanczos(A, y, n)
# display(L)
# display(alpha)
# display(beta)
# 
# println("builtin arnoldi")
# Ks = arnoldi(A, y)
# display(Ks)
