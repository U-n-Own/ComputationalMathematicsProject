using LinearAlgebra
using SparseArrays
using Random

function generate_quadratic_costs(lin_costs)
    m = size(lin_costs)[1]
    quadratic_costs = zeros(m)

    a = 3
    b = 10

    for i in 1:m
        quadratic_costs[i] = rand() * (b - a) * lin_costs[i] + a * lin_costs[i]
    end
    return quadratic_costs
end

function gen_A_from_data(lin_costs, E_bar)
    D = Diagonal(generate_quadratic_costs(lin_costs))
    E = E_bar[1 : size(E_bar)[1] - 1, :]

    Eᵀ = transpose(E) * 1.0

    Z = zeros(size(E, 1), size(E, 1))

    A = [D Eᵀ ;
         E Z]

    return D, E, SparseMatrixCSC(A)

end

function gen_test_prec_from_data(lin_costs, E_bar)
    D = Diagonal(generate_quadratic_costs(lin_costs))
    E = E_bar[1 : size(E_bar)[1] - 1, :]
    Eᵀ = transpose(E) * 1.0
    Dinv = Matrix(inv(D))
    S = E * Dinv * Eᵀ
    Z = zeros(size(D, 1), size(S, 2))
    P = [D Z ;
         E S]

    return SparseMatrixCSC(P)
end

function gen_A_all_ones(E_bar)
    E = E_bar[1 : size(E_bar)[1] - 1, :]
    Deye = I(size(E, 2)) * 1.0
    Eᵀ = transpose(E) * 1.0
    Z = zeros(size(E, 1), size(E, 1))

    A = [Deye Eᵀ ;
        E Z]

    return Deye, E, SparseMatrixCSC(A)
end


function gen_test_prec_all_ones(E_bar)
    E = E_bar[1 : size(E_bar)[1] - 1, :]
    Deye = I(size(E, 2)) * 1.0
    Eᵀ = transpose(E) * 1.0
    S = E * Deye * Eᵀ
    Z = zeros(size(Deye, 1), size(S, 2))

    P = [Deye Z ;
         E S]

    return SparseMatrixCSC(P)
end


function gen_A_uniform(E_bar)
    E = E_bar[1 : size(E_bar)[1] - 1, :]
    D = Diagonal(rand(size(E, 2)))
    Eᵀ = transpose(E) * 1.0
    Z = zeros(size(E, 1), size(E, 1))

    A = [D Eᵀ ;
         E Z]

    return D, E, SparseMatrixCSC(A)
end


function gen_y_from_data(node_flows, lin_costs)
    res = vcat(-lin_costs * 1.0, 1.0 * node_flows[1:size(node_flows)[1] - 1])
    return res
end



