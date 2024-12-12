# This function selects grid points based on various options.
# Inputs:
#   X: Observations matrix (n rows, d columns)
#   grid_option: Selection method for grid points
#                1 - Use data points as grid points
#                2 - Random subsample of the data points
#                3 - Uniform grid in a compact region containing the data
#                4 - Logspace for y, linspace for x
#   m: Number of grid points (optional, determined based on grid_option if not provided)
# Outputs:
#   U: Grid points matrix (m rows, d columns)
#   m: Number of grid points

using Random
using LinearAlgebra

function select_grid(X, grid_option=1, m=nothing)
    n, d = size(X)
    
    if isnothing(grid_option) || grid_option == 1
        grid_option = 1
    end

    if grid_option == 1
        # Case 1: Use all data points as grid points
        U = X
        m = n

    elseif grid_option == 2
        # Case 2: Random subsample of the data points
        m = isnothing(m) ? round(Int, sqrt(n)) : min(m, n)
        indices = randperm(n)[1:m]
        U = X[indices, :]

    elseif grid_option == 3
        # Case 3: Uniform grid in a compact region containing the data
        if isnothing(m)
            m = n
        end
        xmax, xmin = maximum(X[:, 1]), minimum(X[:, 1])
        ymax, ymin = maximum(X[:, 2]), minimum(X[:, 2])
        ratio = 1.0
        my = round(Int, sqrt(m / ratio))
        mx = round(Int, sqrt(m * ratio))
        m = mx * my
        xgrid = range(xmin, xmax, length=mx)
        ygrid = range(ymin, ymax, length=my)
        Xg, Yg = Iterators.product(xgrid, ygrid) |> collect |> permutedims
        U = hcat(Xg, Yg)

    elseif grid_option == 4
        # Case 4: Logspace for y, linspace for x
        if isnothing(m)
            m = n
        end
        xmax, xmin = maximum(X[:, 1]), minimum(X[:, 1])
        ymax, ymin = maximum(X[:, 2]), minimum(X[:, 2])
        ratio = 1.0
        my = round(Int, sqrt(m / ratio))
        mx = round(Int, sqrt(m * ratio))
        m = mx * my
        xgrid = range(xmin, xmax, length=mx)
        ygrid = exp10.(range(log10(ymin), log10(ymax), length=my))
        Xg, Yg = Iterators.product(xgrid, ygrid) |> collect |> permutedims
        U = hcat(Xg, Yg)

    else
        U = X
        m = n
    end

    return U, m
end

## function test(complete)
X = rand(100, 3) 
grid_option = 1
m = 50
U, m_selected = select_grid(X, grid_option, m)
println("U: ", U)
println("Number of grid points: ", m_selected)
