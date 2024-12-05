using LinearAlgebra 
using Random
using SparseArrays
include("Arnoldi.jl")
include("QR.jl")


Random.seed!(42)


function GMRES(A::SparseMatrixCSC, y::Vector, n::Int)
    
    Q, H = Arnoldi(A, y, n)

    v = vcat(I(1), zeros(n))[:] * norm(y)

    z_our = LeastSquares_QR(SparseMatrixCSC(H[:, 1:n]), SparseVector(v))
    
    #z = H[:, 1:n] \ v
    
    x = Q[:, 1:n] * z_our

    return x

end


A = sprand(100, 100, 0.5)

println("Det sparse",det(A))

println(cond(Matrix(A), 2))

y = rand(100)
n = 400

x = GMRES(A, y, n)

println(norm(A*x - y))
 