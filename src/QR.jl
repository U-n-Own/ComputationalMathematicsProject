using LinearAlgebra
using Random
using SparseArrays

Random.seed!(42)

function householder_v(x :: Vector)
    s = norm(x)
    if x[1] >= 0
        s = -s
    end

    v = copy(x)
    v[1] = v[1] - s

    return v / norm(v)
end


function QR_fact(A :: Matrix)
    m, n = size(A)
    Q = Matrix(1.0 * I(m))
    R = copy(A)

    if m < n
        throw(ArgumentError("Number of rows of the matrix in QR must be greater than or equal to number of columns"))
    end

    for i = 1:n
        v = R[i:end, i]
        u = householder_v(v)
        H = (1.0 * I - 2*u*(u'))
        R[i:end, i:end] = H * R[i:end, i:end]

        QH = Q[:, i:end] * H

        Q[:, i:end] = QH
    end

    return (Q, R)
end

function LeastSquares_QR(Q :: SparseMatrixCSC, R :: SparseMatrixCSC, b :: SparseVector)
    """
    Solve Least Squares problems given QR factorization by back substitution
    """

    Qb = Q' * b
    
    n = size(R)[2]
    
    x = zeros(n)
    
    # We go backward since we have a triangual matrix the last is just a scalar assignment and then 
    # we back subtitute the result in the row above...
    
    for i = n:-1:1
        #general formula is : x_i = (y_i - sum_{j=i+1}^{n} u_{ij}x_j)/u_{ii} taken from https://algowiki-project.org/en/Backward_substitution
        x[i] = (Qb[i] - dot(R[i, i+1:end], x[i+1:end])) / R[i, i]
    end

    return x
end

#v = [2.1297 ; 0.6333]
#= v = rand(9)
A = rand(9, 9)
Q, R = QR_fact(A)
display(Q)
display(R)

x = LeastSquares_QR(SparseMatrixCSC(Q), SparseMatrixCSC(R), SparseVector(v))

println(norm(A*x - v)) =#