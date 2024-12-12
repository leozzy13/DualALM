# EM algorithm for solving the NPMLE with unknown supports in Julia
# Inputs:
#   X: observations (n x d) matrix
#   SIGMA: diagonal covariance matrices (n x d)
#   m: number of grid points
#   options: dictionary containing optional parameters like 'supps_initial' and 'stoptol'
# Outputs:
#   x: estimated mixture proportions
#   supps: estimated support points
#   hist: dictionary containing the objective history
#   k: number of iterations performed

using Random
using LinearAlgebra

function EM(X::Matrix{Float64}, SIGMA::Matrix{Float64}, m::Int, options::Dict=Dict())
    println("\n----------------- EM algorithm--------------------")
    maxiter = 100
    stoptol = 1e-4
    n, d = size(X)

    x = fill(1 / m, m)
    

    if m < n
        supps = X[randperm(n)[1:m], :]
    elseif m == n
        supps = copy(X)
    end

    if haskey(options, "supps_initial")
        supps = options["supps_initial"]
    end
    if haskey(options, "stoptol")
        stoptol = options["stoptol"]
    end

    Sigma = reshape(SIGMA', d, n)'
    inv_Sigma = 1.0 ./ Sigma

    inv_SigmaX = inv_Sigma .* X

    hist = Dict("obj" => Float64[]) 

    obj_old = -Inf
    k_final = 0

    for k in 1:maxiter
        k_final = k
        # L matrix
        L, _, _ = likelihood_matrix(X, supps, SIGMA)

        # E-step
        Lx = L * x
        gamma_hat = L .* (ones(n) * x') ./ (Lx * ones(1, m))

        # M-step
        supps = (gamma_hat' * inv_SigmaX) ./ (gamma_hat' * inv_Sigma)
        x = sum(gamma_hat, dims=1)' / n
        obj = sum(log.(Lx)) / n
        push!(hist["obj"], obj)

        println("iter = $(k), log-likelihood = $(round(obj, digits=8))")

        if k > 1 && obj - obj_old < stoptol
            break
        end
        obj_old = obj
    end

    return x, supps, hist, k_final
end



X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
SIGMA = [0.1 0.2; 0.3 0.4; 0.5 0.6]
m = 2
options = Dict("supps_initial" => [1.5 2.5; 4.5 5.5])
x, supps, hist, k = EM(X, SIGMA, m, options)


println("x:")
println(x)
println("supps:")
println(supps)
println("hist:")
println(hist)
println("k:")
println(k)

