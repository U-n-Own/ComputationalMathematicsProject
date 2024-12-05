using LinearAlgebra

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

M = rand(Float64, (4, 4))
println("M")
display(M)
Q, R = QR_fact(M)
println("final Q")
display(Q)
println("inverse of Q")
display(inv(Q))
println("final R")
display(R)
println("final Q * R")
display(Q*R)
println("initial M")
display(M)



