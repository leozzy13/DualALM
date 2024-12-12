using LinearAlgebra
using SparseArrays

"""
    mexbwsolve(Rt, b)

Solve R * x = b where R is upper triangular and Rt = transpose(R).
Returns the solution vector x.
"""
function mexbwsolve(R, b)
    if size(R, 1) != size(R, 2)
        throw(DimensionMismatch("R should be a square matrix"))
    end

    if !istril(R)
        throw(ArgumentError("R must be lower triangular"))
    end

    Rt_upper = UpperTriangular(R')

    x = Rt_upper \ b

    return x
end


## function test(complete)
n = 10
Rt = SparseMatrixCSC(tril(ones(n, n) .+ 0.7 * I(n)))
b = collect(1:n)
x = mexbwsolve(Rt, b)
print(x)