using LinearAlgebra
using Random
using SparseArrays

# Random.seed!(42)

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
        if j % 500 == 0
            println(j, "-th Lanczos iteration")
        end
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
