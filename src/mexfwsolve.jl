"""
    mexfwsolve(Rt, b)

Solve R * x = b where R is lower triangular and Rt = transpose(R).
Returns the solution vector x.
"""

using LinearAlgebra
using SparseArrays

function mexfwsolve(R::SparseMatrixCSC, b::AbstractVector)
    if size(R, 1) != size(R, 2)
        throw(DimensionMismatch("R should be a square matrix"))
    end

    if !istriu(R)
        throw(ArgumentError("R must be upper triangular"))
    end

    for i in 1:size(R, 1)
        if R[i, i] == 0
            throw(SingularException(i))
        end
    end

    Rt_lower = LowerTriangular(R')

    x = Rt_lower \ b

    return x
end



## function test(complete)
n = 10
Rt = SparseMatrixCSC(triu(ones(n, n) .+ 0.7 * I(n)))
b = collect(1:n)
x = mexfwsolve(Rt, b)
print(x)