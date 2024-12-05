using LinearAlgebra
using Random
using SparseArrays

Random.seed!(42)

function Arnoldi(A::SparseMatrixCSC, y::Vector, n::Int)
    """
    Generate basis q1,...,q{n+1} in K_{n+1}(A, y)
    - A is a matrix
    - y is a vector
    - n is the number of vectors to generate
    - Arnoldi copied from Poloni
    Returns:
    Q = [q1, q2, ..., q{n+1}]
    """
    Q = zeros(size(A, 1), n+1)
    H = zeros(n+1, n)
    q1 = y/norm(y)

    # first vector of the basis
    Q[1:end, 1] = q1

    for j in 1:n

        # extend basis from the past j to get one more
        # w = Aqj
        w = A*Q[1:end, j] 

        for i in 1:j
            
            # h1 scalar product q1^Tw
            H[i,j] = Q[1:end, i]'*w
            # Keep subtracting multiples of the q_i to make it orthogonal
            w = w - Q[1:end, i] * H[i,j]
        end
    
    H[j+1, j] = norm(w)
    # we just add the new vector to the basis
    Q[1:end, j+1] = w/H[j+1, j]
    
    end

    return (Q, H)

end
