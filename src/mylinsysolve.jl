# Function to solve linear systems using Cholesky factorization 
# Arguments:
#   L  : Structure containing Cholesky factorization information.
#   r  : Right-hand side vector of the linear system.
#
# Returns:
#   q  : Solution vector.

function mylinsysolve(L, r)
    q = similar(r)
    if L[:matfct_options] == "chol"
        # Solve using standard Cholesky factorization
        q[L[:perm]] = mextriang(L[:R], mextriang(L[:R], r[L[:perm]], 2), 1)
    elseif L[:matfct_options] == "spcholmatlab"
        # Solve using sparse Cholesky factorization
        q[L[:perm]] = mexbwsolve(L[:Rt], mexfwsolve(L[:R], r[L[:perm]]))
    end
    return q
end
