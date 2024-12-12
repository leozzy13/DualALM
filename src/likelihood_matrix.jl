using Distributions

"""
Generate the matrix L

# Arguments
- `X::Array{Float64, 2}`: Observations (n x d)
- `U::Array{Float64, 2}`: Grid points (m x d)
- `SIGMA::Union{Array{Float64, 3}, Array{Float64, 2}}`: Covariance (1 x d x n or d x d)
- `normalizerows::Bool`: Whether to normalize rows (default: false)
- `restrict_dist::Bool`: Whether to restrict distance (default: false)

# Returns
- `L::Array{Float64, 2}`: Likelihood matrix (n x m)
- `rowmax::Vector{Float64}`: Maximum value of each row of L
- `removeind::Vector{Int}`: Index set of removed observations
"""

function likelihood_matrix(X, U, SIGMA, normalizerows=false, restrict_dist=false)
    n, d = size(X)
    m, _ = size(U)
    L = zeros(n, m)
    rowmax = zeros(n)
    tiny = restrict_dist ? 1e-9 : 1e-150
    cnt = 0
    removeind = Int[]
    sz = ndims(SIGMA)

    for i in 1:n
        XI = X[i, :]
        if sz == 3
            SIG =Diagonal(vec(SIGMA[:, :, i]))
        elseif sz == 2
            SIG = SIGMA
        else
            SIG = []
        end

        tmp = [pdf(MvNormal(U[j, :] .- XI, SIG), zeros(d)) for j in 1:m]
        maxtmp = maximum(tmp)

        if maxtmp > tiny
            cnt += 1
            rowmax[cnt] = maxtmp
            if normalizerows
                L[cnt, :] .= max.(tmp, tiny) ./ maxtmp
            else
                L[cnt, :] .= max.(tmp, tiny)
            end
        else
            push!(removeind, i)
        end
    end

    L = L[1:cnt, :]
    rowmax = rowmax[1:cnt]
    return L, rowmax, removeind
end





## function test(complete for SIGMA in dimension 2 and 3)
X = [1.0 2.0; 3.0 4.0]
U = [0.5 1.5; 2.5 3.5; 4.5 5.5]
SIGMA = cat([1, 2], [3, 4], [5, 6], [7, 8]; dims=3)
L, rowmax, removeind = likelihood_matrix(X, U, SIGMA, false, false)

println("L:")
println(L)
println("rowmax:")
println(rowmax)
println("removeind:")
println(removeind)




