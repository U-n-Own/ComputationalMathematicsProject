using LinearAlgebra
using Random
using SparseArrays
using ExponentialUtilities  # to compare with builtin arnoldi
using TickTock  # to test complexity of LanczosQR

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
    
    Q = zeros(n+1, n+1)
    R = zeros(n+1, n)
    
    
    
    
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
            w = A * q               # O(m^2)
            alpha[1] = w' * q       # O(m)
            w = w - alpha[1] * q    # O(m)
            beta[1] = norm(w)       # O(m)
        else
            # inductive case
            q = w / beta[j-1]       # O(m)
            L[:, j] = q             # O(m)
            
            w = A * q               # O(m^2)
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

#=
D = SparseMatrixCSC(1.0 * I(100))
E = sprand(20, 100, .7)
Z = SparseMatrixCSC(zeros(20, 20))

A = hcat(vcat(D, E), vcat(E', Z))

b = sprand(120, .7)

L, alpha, beta, Q, R = LanczosQR(A, b, 120)

H = Q*R
println(size(R))

e1 = Matrix(1.0 * I(121))[:, 1]

x = L * (H \ (e1 * norm(b)))

println(norm(A * x - b))
=#
