# This function solves a linear system using different solvers based on the problem size and structure.
# Inputs:
#   rhs: right-hand side of the linear system
#   LL: matrix or struct representing the factorization or system matrix
#   prox_v1_prime_m: diagonal elements used in the linear system
#   v2: indicator vector for active elements
#   par: parameters including n, m, sigma, approxL, etc.
# Outputs:
#   dv: solution of the linear system
#   resnrm: residual norm (if applicable)
#   solve_ok: flag indicating if the solver was successful
#   par: updated parameters (with r value)

using LinearAlgebra
using Random

using LinearAlgebra

function Linsolver_MLE(rhs, LL, prox_v1_prime_m, v2, par)
    n = par[:n]
    m = par[:m]
    sigma = par[:sigma]
    
    J = v2 .> 0
    r = sum(J)
    par[:r] = r

    solveby = "pcg"
    if n <= 5000
        solveby = "pdirect"
    end
    if (r < 2000) || (n > 5000 && r < 5000)
        solveby = "ddirect"
    end

    if solveby == "pdirect"
        if par[:approxL]
            U = LL[:U]
            V = LL[:V]
            VJ = V[J, :]
            LLT = U * (VJ' * VJ) * U'
            for i in 1:n
                LLT[i, i] += prox_v1_prime_m[i]
            end
            cholLLT = mycholAAt(LLT, n)
        else
            LJ = LL[:matrix][:, J]
            LLT = LJ * LJ'
            for i in 1:n
                LLT[i, i] += prox_v1_prime_m[i]
            end
            cholLLT = mycholAAt(LLT, n)
        end
        dv = mylinsysolve(cholLLT, rhs * (n^2 / sigma))
        resnrm = 0
        solve_ok = 1

    elseif solveby == "pcg"
        if par[:approxL]
            U = LL[:U]
            V = LL[:V]
            VJ = V[J, :]
            Afun = v -> ((prox_v1_prime_m .* v + U * ((VJ * (v' * U)')' * VJ)') * (sigma / n^2))
        else
            LJ = LL[:matrix][:, J]
            Afun = v -> ((prox_v1_prime_m .* v + LJ * (LJ' * v)) * (sigma / n^2))
        end
        dv, _, resnrm, solve_ok = psqmr(Afun, rhs, par)

    elseif solveby == "ddirect"
        rhs = rhs * (n^2 / sigma)
        prox_v1_prime_m .= prox_v1_prime_m .+ eps()

        if par[:approxL]
            U = LL[:U]
            V = LL[:V]
            VJ = V[J, :]
            # Compute rhstmp
            rhstmp = VJ * ((rhs ./ prox_v1_prime_m)' * U)'
            # Compute LTL
            LTL = VJ * (U' * (U ./ prox_v1_prime_m)) * VJ'
            for i in 1:r
                LTL[i, i] += 1
            end
            if r <= 1000
                dv = LTL \ rhstmp
            else
                cholLTL = mycholAAt(LTL, r)
                dv = mylinsysolve(cholLTL, rhstmp)
            end
            dv = (U * (dv' * VJ)') ./ prox_v1_prime_m
            dv = rhs ./ prox_v1_prime_m .- dv
        else
            if r == m
                LJ = LL[:matrix]
            else
                LJ = LL[:matrix][:, J]
            end
            LJ2 = LJ ./ prox_v1_prime_m
            rhstmp = (rhs' * LJ2)'
            LTL = LJ' * LJ2
            LTL = I(r) + LTL
            if r <= 1000
                dv = LTL \ rhstmp
            else
                cholLTL = mycholAAt(LTL, r)
                dv = mylinsysolve(cholLTL, rhstmp)
            end
            dv = LJ2 * dv
            dv = rhs ./ prox_v1_prime_m .- dv
        end
        resnrm = 0
        solve_ok = 1
    end

    return dv, resnrm, solve_ok, par
end




## function test(complete)
n, m = 10, 5
sigma = 0.1
rhs = [1.0, 0.5, -0.2, 0.3, 0.8, -0.6, 0.9, -1.1, 0.4, -0.7]
prox_v1_prime_m = [0.2, 0.1, 0.3, 0.4, 0.5, 0.3, 0.6, 0.7, 0.2, 0.1]
v2 = [0.1, -0.3, 0.5, -0.2, 0.4]
par = Dict("n" => n, "m" => m, "sigma" => sigma, "approxL" => false)

LL = (U = [0.5 0.2 0.1 0.3 0.4; 0.2 0.3 0.5 0.1 0.4; 0.6 0.1 0.3 0.7 0.2; 0.3 0.6 0.4 0.5 0.1; 0.1 0.5 0.2 0.3 0.6; 0.4 0.7 0.3 0.1 0.5; 0.6 0.3 0.2 0.5 0.7; 0.5 0.4 0.6 0.3 0.2; 0.7 0.2 0.5 0.1 0.3; 0.4 0.6 0.3 0.7 0.5],
     V = [0.2 0.3 0.5 0.1 0.4; 0.3 0.6 0.4 0.5 0.2; 0.5 0.1 0.3 0.7 0.6; 0.1 0.4 0.5 0.2 0.3; 0.4 0.2 0.6 0.3 0.7],
     matrix = [0.5 0.2 0.3 0.1 0.4; 0.6 0.3 0.5 0.7 0.2; 0.4 0.1 0.3 0.6 0.5; 0.3 0.7 0.2 0.5 0.1; 0.5 0.4 0.6 0.3 0.2; 0.7 0.5 0.3 0.2 0.4; 0.6 0.1 0.4 0.3 0.7; 0.2 0.3 0.5 0.6 0.1; 0.4 0.2 0.1 0.7 0.5; 0.3 0.6 0.2 0.4 0.1])
x, resnrm, solve_ok, par_updated = Linsolver_MLE(rhs, LL, prox_v1_prime_m, v2, par)

println("x: ", x)
println("resnrm: ", resnrm)
println("solve_ok: ", solve_ok)
println("par_updated: ", par_updated)


