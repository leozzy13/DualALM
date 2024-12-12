function mextriang(U, b, options::Int = 1)

    if size(U, 1) != size(U, 2)
        throw(ArgumentError("Matrix U must be square"))
    end
    if size(U, 1) != length(b)
        throw(ArgumentError("Dimension of b must match the number of rows of U"))
    end
    if options != 1 && options != 2
        throw(ArgumentError("Options must be either 1 or 2"))
    end
    if !istriu(U)
        throw(ArgumentError("Matrix U must be upper triangular"))
    end

    if options == 1
        # Solve U * y = b
        return U \ b
    elseif options == 2
        # Solve U' * y = b
        return U' \ b
    end
end



## function test(complete)
n = 17
Rt = triu(ones(n, n) .+ 0.8 * I(n))
b = collect(1:n)
x = mextriang(Rt, b, 2)
print(x)