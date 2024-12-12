using LinearAlgebra
using SparseArrays

"""
    mexbwsolve(Rt, b)

Solve R * x = b where R is upper triangular and Rt = transpose(R).
Returns the solution vector x.
"""
function mexbwsolve(R::SparseMatrixCSC, b::AbstractVector)
    if size(R, 1) != size(R, 2)
        throw(DimensionMismatch("R should be a square matrix"))
    end

    if !istril(R)
        throw(ArgumentError("R must be lower triangular"))
    end

    for i in 1:size(R, 1)
        if R[i, i] == 0
            throw(SingularException(i))
        end
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