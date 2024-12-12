# This function finds the step size for an optimization algorithm using backtracking line search.
# Inputs:
#   Grad: gradient vector
#   dv, LTdv: direction vectors
#   LL: linear operator
#   phi0: initial phi value
#   v1input0, prox_v10, prox_v1_prime_m0: initial v1 values and proximal data
#   v2input0, prox_v20, Lprox_v20: initial v2 values and proximal data
#   tol: tolerance for convergence
#   options: search options
#   par: dictionary of parameters
# Outputs:
#   phi: updated phi value
#   v1input, prox_v1, prox_v1_prime_m: updated v1 and proximal data
#   v2input, prox_v2, Lprox_v2: updated v2 and proximal data
#   alp: step size
#   iter: number of iterations
#   par: updated parameters

function findstep(Grad, dv, LTdv, LL, phi0, v1input0, prox_v10, prox_v1_prime_m0, v2input0, prox_v20, Lprox_v20, tol, options, par)
    sigma = par[:sigma]
    n = par[:n]
    m = par[:m]
    printyes = par[:printyes]
    maxit = ceil(Int, log(1 / (tol + eps())) / log(2))
    c1 = 1e-4
    c2 = 0.9
    g0 = -dot(Grad, dv)

    if g0 <= 0
        if printyes
            println("\n Need an ascent direction, ", g0)
        end
        return phi0, v1input0, prox_v10, prox_v1_prime_m0, v2input0, prox_v20, Lprox_v20, 0.0, 0, par
    end

    phi = phi0
    alp = 1.0
    alpconst = 0.5
    LB, UB = 0.0, 1.0
    gLB, gUB = 0.0, 0.0
    v1input = copy(v1input0)
    v2input = copy(v2input0)
    prox_v1 = copy(prox_v10)
    prox_v2 = copy(prox_v20)
    prox_v1_prime_m = copy(prox_v1_prime_m0)
    iter = 0

    for i in 1:maxit
        iter = i
        if iter == 1
            alp = 1.0
            LB = 0.0
            UB = 1.0
        else
            alp = alpconst * (LB + UB)
        end

        v1input = v1input0 .+ alp .* dv
        v2input = v2input0 .+ (alp / n) .* LTdv
        prox_v1, M_v1, _, prox_v1_prime_m = prox_h(v1input, sigma / (n^2))
        prox_v2 = max.(v2input, 0.0)  # Reassign prox_v2, not mutate
        phi = -(M_v1 + (sigma / 2) * norm(prox_v2)^2)
        tmp = (sigma / (n^2)) * (v1input .- prox_v1)
        galp = -dot(tmp, dv) - (sigma / n) * dot(prox_v2, LTdv)

        if iter == 1
            gLB, gUB = g0, galp
            if sign(gLB) * sign(gUB) > 0
                if printyes
                    print("|")
                end
                break
            end
        end

        if (abs(galp) < c2 * abs(g0)) && (phi - phi0 - c1 * alp * g0 > eps())
            if (options == 1) || ((options == 2) && (abs(galp) < tol))
                if printyes
                    print(":")
                end
                break
            end
        end

        if sign(galp) * sign(gUB) < 0
            LB, gLB = alp, galp
        elseif sign(galp) * sign(gLB) < 0
            UB, gUB = alp, galp
        end
    end

    Lprox_v2 = LL[:times](prox_v2)
    par[:count_L] += 1

    if printyes && (iter == maxit)
        print("m")
    end

    return phi, v1input, prox_v1, prox_v1_prime_m, v2input, prox_v2, Lprox_v2, alp, iter, par
end








## function test(complete)
Grad = [-1.0, -0.5, -0.3, -0.2, -0.1]
dv = [0.4, -0.3, 0.2, 0.1, 0.5]
LTdv = [0.3, 0.2, -0.1, 0.4, -0.2]
LL = Dict(:times => x -> 2.0 .* x)
phi0 = 0.5
v1input0 = [0.6, -0.4, 0.3, 0.2, -0.5]
prox_v10 = [0.7, -0.3, 0.1, 0.5, -0.2]
prox_v1_prime_m0 = [0.2, 0.3, 0.1, 0.4, 0.5]
v2input0 = [0.3, 0.5, -0.1, 0.6, 0.2]
prox_v20 = [0.4, 0.1, 0.2, 0.3, 0.5]
Lprox_v20 = [0.2, 0.4, 0.6, 0.8, 1.0]
tol = 1e-10
options = 2
par = Dict(:sigma => 0.1, :n => 5, :m => 5, :printyes => true, :count_L => 0)


phi, v1inpu, prox_v1, prox_v1_prime_m, v2input, prox_v2, Lprox_v2, alp, iter, par_updated = findstep(Grad, dv, LTdv, LL, phi0, v1input0, prox_v10, prox_v1_prime_m0, v2input0, prox_v20, Lprox_v20, tol, options, par)


println("phi: ", phi)
println("v1input: ", v1input)
println("prox_v1: ", prox_v1)
println("prox_v1_prime_m: ", prox_v1_prime_m)
println("v2input: ", v2input)
println("prox_v2: ", prox_v2)
println("Lprox_v2: ", Lprox_v2)
println("alp: ", alp)
println("iter: ", iter)
println("par_updated: ", par_updated)



