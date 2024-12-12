# This function performs Cholesky decomposition with checks for sparse matrices
# Inputs:
#   ATT: matrix to decompose
#   m: size of the matrix
# Outputs:
#   L: struct containing decomposition results

using SparseArrays

function mycholAAt(ATT, m)
    use_spchol = count(!iszero, ATT) < 0.2 * m * m
    L = Dict{Symbol,Any}()

    if use_spchol
        ch = cholesky(sparse(ATT); check=false)
        L[:R] = ch.U        # Upper triangular factor
        L[:Rt] = (ch.U)'    # Transpose of R
        L[:p] = ch.info     # Similar to L.p in MATLAB
        L[:perm] = ch.p     # Permutation vector
        L[:matfct_options] = "spcholmatlab"
    else
        if issparse(ATT)
            ATT = Matrix(ATT)
        end
        ch = cholesky(ATT; check=false)
        L[:matfct_options] = "chol"
        L[:perm] = collect(1:m)
        L[:R] = ch.U  
    end

    return L
end




## function test(complete)
m = 5
ATT = [3.0 1.0 0.0 0.0 0.0;
       1.0 3.0 1.0 0.0 0.0;
       0.0 1.0 2.0 1.0 0.0;
       0.0 0.0 1.0 3.0 1.0;
       0.0 0.0 0.0 1.0 2.0]

L = mycholAAt(ATT, m)
println("Cholesky decomposition result L: ")
println(L)

m = 6
ATT = [3.0 0.0 0.0 0.0 0.0 0.0;
       0.0 3.0 0.0 0.0 0.0 0.0;
       0.0 0.0 2.0 0.0 0.0 0.0;
       0.0 0.0 0.0 2.0 0.0 0.0;
       0.0 0.0 0.0 0.0 2.0 0.0;
       0.0 0.0 0.0 0.0 0.0 1.0]
L_sparse = mycholAAt(ATT, m)
println("Cholesky decomposition result L")
println(L_sparse)
