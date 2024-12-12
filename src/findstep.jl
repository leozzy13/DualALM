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

using LinearAlgebra


function findstep(Grad, dv, LTdv, LL, phi0, v1input0, prox_v10, prox_v1_prime_m0, 
                  v2input0, prox_v20, Lprox_v20, tol, options, par)

    sigma = par[:sigma]
    n = par[:n]
    # m = par[:m] # m is not used in this function
    printyes = par[:printyes]

    maxit = ceil(Int, log(1 / (tol + eps())) / log(2))
    c1 = 1e-4
    c2 = 0.9
    g0 = -dot(Grad, dv)

    # If no ascent direction
    if g0 <= 0
        if printyes > 0
            @printf("\n Need an ascent direction, %2.1e  ", g0)
        end    
        phi = phi0
        v1input = copy(v1input0)
        prox_v1 = copy(prox_v10)
        prox_v1_prime_m = copy(prox_v1_prime_m0)
        v2input = copy(v2input0)
        prox_v2 = copy(prox_v20)
        Lprox_v2 = copy(Lprox_v20)
        alp = 0.0
        iter = 0
        return phi, v1input, prox_v1, prox_v1_prime_m, v2input, prox_v2, Lprox_v2, alp, iter, par
    end

    alpconst = 0.5
    # Initialize iteration variables
    v1input = copy(v1input0)
    v2input = copy(v2input0)
    prox_v1 = copy(prox_v10)
    prox_v2 = copy(prox_v20)
    prox_v1_prime_m = copy(prox_v1_prime_m0)

    # Line search initialization
    alp = 1.0
    LB = 0.0
    UB = 1.0
    gLB = 0.0
    gUB = 0.0

    iter = 0
    for i in 1:maxit
        iter = i
        if iter == 1
            alp = 1.0
            LB = 0.0
            UB = 1.0
        else
            alp = alpconst*(LB + UB)
        end

        v1input = v1input0 .+ alp .* dv
        v2input = v2input0 .+ (alp / n) .* LTdv

        prox_v1, M_v1, _, prox_v1_prime_m = prox_h(v1input, sigma/(n^2))
        prox_v2 .= max.(v2input, 0.0)  

        phi = -(M_v1 + (sigma/2)*norm(prox_v2)^2)

        tmp = (sigma/(n^2))*(v1input .- prox_v1)
        galp = -dot(tmp, dv) - (sigma/n)*dot(prox_v2, LTdv)

        if iter == 1
            gLB = g0
            gUB = galp
            # Check if gLB and gUB have same sign
            if sign(gLB)*sign(gUB) > 0
                if printyes > 0
                   print("|")
                end
                break
            end
        end

        # Armijo and curvature conditions: 
        #   Condition: (abs(galp) < c2*abs(g0)) and (phi > phi0 + c1*alp*g0 + eps())
        if (abs(galp) < c2*abs(g0)) && (phi > phi0 + c1*alp*g0 + eps())
            if (options == 1) || ((options == 2) && (abs(galp) < tol))
                if printyes > 0
                   print(":")
                end
                break
            end
        end

        # Update LB and UB based on galp
        if sign(galp)*sign(gUB) < 0
            LB = alp; gLB = galp
        elseif sign(galp)*sign(gLB) < 0
            UB = alp; gUB = galp
        end
    end

    # Apply the linear operator again after determining alp
    Lprox_v2 = LL[:times](prox_v2)
    par[:count_L] = par[:count_L] + 1

    if printyes > 0 && (iter == maxit)
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



