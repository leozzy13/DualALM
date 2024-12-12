# Function to solve linear systems using Cholesky factorization 
# Arguments:
#   L  : Structure containing Cholesky factorization information.
#   r  : Right-hand side vector of the linear system.
#
# Returns:
#   q  : Solution vector.

function mylinsysolve(L, r)
    q = similar(r)  # allocate a vector q of the same size as r
    perm = L[:perm]

    if L[:matfct_options] == "chol"
        q[perm] = mextriang(L[:R], mextriang(L[:R], r[perm], 2), 1)

    elseif L[:matfct_options] == "spcholmatlab"
        q[perm] = mexbwsolve(L[:Rt], mexfwsolve(L[:R], r[perm]))

    else
        error("Unknown matfct_options")
    end

    return q
end

