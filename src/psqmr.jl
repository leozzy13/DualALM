# Preconditioned Symmetric QMR Solver
# Inputs:
#   matvecfname: Function to compute matrix-vector products
#   b: Right-hand side vector
#   par: Parameter dictionary
#   x0: Initial guess for the solution (optional)
#   Ax0: Initial matrix-vector product (optional)
# Outputs:
#   x: Solution vector
#   Ax: Matrix-vector product corresponding to the solution
#   resnrm: Norm of residual vector at each iteration
#   solve_ok: Status of convergence

using LinearAlgebra

function psqmr(matvecfun, b, par; x0=nothing, Ax0=nothing)
    N = length(b)
    maxit = max(5000, round(Int, sqrt(N)))
    tol = 1e-6*norm(b)
    stagnate_check = 20
    miniter = 0

    if haskey(par, :maxit)
        maxit = par[:maxit]
    end
    if haskey(par, :tol)
        tol = par[:tol]
    end
    if haskey(par, :stagnate_check_psqmr)
        stagnate_check = par[:stagnate_check_psqmr]
    end
    if haskey(par, :minitpsqmr)
        miniter = par[:minitpsqmr]
    end

    if x0 === nothing
        x0 = zeros(N)
    end

    solve_ok = 1
    printlevel = 1

    x = x0
    if norm(x) > 0
        Aq = Ax0 === nothing ? matvecfun(x0) : Ax0
    else
        Aq = zeros(N)
    end
    r = b - Aq
    err = norm(r)
    resnrm = [err]
    minres = err

    q = r
    tau_old = norm(q)
    rho_old = dot(r, q)
    theta_old = 0.0
    d = zeros(N)
    res_vec = r
    Ad = zeros(N)

    tiny = -1e-30
    iter = 0

    for i in 1:maxit
        iter = i
        Aq = matvecfun(q)
        sigma = dot(q, Aq)
        if abs(sigma) < abs(tiny)
            solve_ok = 2
            if printlevel > 0
                print("s1")
            end
            break
        else
            alpha = rho_old / sigma
            r = r - alpha * Aq
        end

        u = r

        theta = norm(u) / tau_old
        c = 1 / sqrt(1 + theta^2)
        tau = tau_old * theta * c
        gam = c^2 * (theta_old^2)
        eta = (c^2 * alpha)
        d = gam * d + eta * q
        x = x + d

        Ad = gam * Ad + eta * Aq
        res_vec = res_vec - Ad
        err = norm(res_vec)
        push!(resnrm, err)
        if err < minres
            minres = err
        end
        if (err < tol) && (iter > miniter) && (dot(b, x) > 0)
            break
        end

        if (iter > stagnate_check) && (iter > 10)
            ratio = resnrm[iter-9:iter+1] ./ resnrm[iter-10:iter]
            if (minimum(ratio) > 0.997) && (maximum(ratio) < 1.003)
                if printlevel > 0
                    print("s")
                end
                solve_ok = -1
                break
            end
        end

        if abs(rho_old) < abs(tiny)
            solve_ok = 2
            print("s2")
            break
        else
            rho = dot(r, u)
            beta = rho / rho_old
            q = u + beta * q
        end

        rho_old = rho
        tau_old = tau
        theta_old = theta
    end

    if iter == maxit
        solve_ok = -2
    end

    Ax = b - res_vec
    return x, Ax, resnrm, solve_ok
end


## function test(complete)
function matvecfname_example(x)
    A = [3.0 2.0; 2.0 6.0]
    return A * x
end

b = [2.0, -83.0]
par = Dict(:tol => 1e-6, :maxit => 100)
x, Ax, resnrm, solve_ok = psqmr(matvecfname_example, b, par)

println("x: ", x)
println("Ax: ", Ax)
println("resnrm: ", resnrm)
println("solve_ok: ", solve_ok)
