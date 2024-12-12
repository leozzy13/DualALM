using Printf

function DualALM_main(
    LL,
    parmain,
    x::Vector{Float64},
    y::Vector{Float64},
    u::Vector{Float64},
    v::Vector{Float64}
)

    tstart = parmain[:tstart]
    stoptol = parmain[:stoptol]
    stopop = parmain[:stopop]
    printyes = parmain[:printyes]
    maxiter = parmain[:maxiter]
    approxL = parmain[:approxL]
    approxRank = parmain[:approxRank]
    sigma = parmain[:sigma]
    m = parmain[:m]
    n = parmain[:n]

    stop = false
    sigmamax = 1e7
    sigmamin = 1e-8
    count_L = 0
    count_LT = 0

    # Initial objective values and feasibilities
    Lx = LL[:times](x)
    count_L += 1
    obj_prim = sum(x) - sum(log.(Lx))/n - 1.0
    obj_dual = sum(log.(u))/n
    obj = [obj_prim, obj_dual]

    relgap = abs(obj_prim - obj_dual) / (1.0 + abs(obj_prim) + abs(obj_dual))
    Rp = Lx .- y
    normy = norm(y, 2)
    primfeas = max(norm(Rp, 2)/normy, norm(min.(x, 0.0), 2)/norm(x, 2))

    LTv = LL[:trans](v) # Use v here, as in MATLAB
    count_LT += 1
    Rd = max.(LTv .- n, 0.0)
    normu_val = norm(u, 2)
    dualfeas = max(norm(Rd, 2)/n, norm(u .- v, 2)/max(normu_val,1e-10))
    maxfeas = max(primfeas, dualfeas)
    eta = norm(y .- 1.0./v, 2)/normy

    if printyes > 0
        println("\n (dimension: m = $m, n = $n, tol = $stoptol)")
        println("---------------------------------------------------")
        println("\n*********************************************" *
                "*******************************************************")
        @printf("\n %5d| [%3.2e %3.2e %3.2e] %- 3.2e| %- 8.6e  %- 8.6e |",
                0, primfeas, dualfeas, eta, relgap, obj_prim, obj_dual)
        @printf(" %5.1f| %3.2e|", (time() - tstart), sigma)
    end

    parNCG = Dict{Symbol, Any}(
        :tolconst => 0.5,
        :count_L => count_L,
        :count_LT => count_LT,
        :approxL => approxL,
        :approxRank => approxRank,
        :m => m,
        :n => n,
        :iter => 0,
        :sigma => sigma
    )

    maxitersub = 20
    ssncgop = Dict(
        :tol => stoptol,
        :printyes => printyes,
        :maxitersub => maxitersub
    )

    runhist = Dict{Symbol, Any}(
        :primfeas => zeros(Float64, maxiter),
        :dualfeas => zeros(Float64, maxiter),
        :sigma    => zeros(Float64, maxiter),
        :primobj  => zeros(Float64, maxiter),
        :dualobj  => zeros(Float64, maxiter),
        :gap      => zeros(Float64, maxiter),
        :relgap   => zeros(Float64, maxiter),
        :ttime    => zeros(Float64, maxiter),
        :ttimessn => zeros(Float64, maxiter),
        :itersub  => zeros(Int, maxiter),
        :iterCG   => zeros(Int, maxiter)
    )

    termination = ""
    for iter in 1:maxiter
        parNCG[:iter] = iter
        parNCG[:sigma] = sigma

        # Adjust maxitersub based on dualfeas
        if dualfeas < 1e-5
            maxitersub = max(maxitersub,35)
        elseif dualfeas < 1e-3
            maxitersub = max(maxitersub,30)
        elseif dualfeas < 1e-1
            maxitersub = max(maxitersub,30)
        end
        ssncgop[:maxitersub] = maxitersub

        tstart_ssn = time()
        x, y, u, v, Lx, LTv, parNCG, _, info_NCG = MLE_SSNCG(LL, x, y, v, LTv, parNCG, ssncgop)
        ttimessn = time() - tstart_ssn

        if info_NCG[:breakyes] < 0
            parNCG[:tolconst] = max(parNCG[:tolconst]/1.06, 1e-3)
        end

        # Compute KKT residual
        Rp = Lx .- y
        normy = norm(y, 2)
        primfeas = max(norm(Rp, 2)/normy, norm(min.(x,0.0), 2)/norm(x,2))
        Rd = max.(LTv .- n, 0.0)
        normu_val = norm(u,2)
        dualfeas = max(norm(Rd,2)/n, norm(u .- v,2)/max(normu_val,1e-10))
        maxfeas = max(primfeas, dualfeas)
        eta = norm(y .- 1.0./v,2)/normy

        primobj = sum(x) - sum(log.(Lx))/n - 1.0
        dualobj = sum(log.(u))/n
        gap = primobj - dualobj
        relgap = abs(gap)/(1.0 + abs(primobj) + abs(dualobj))

        # Check stopping conditions
        stop = false
        if (stopop == 1) && (maxfeas < stoptol) && (eta < stoptol)
            stop = true
        elseif (stopop == 2) && (eta < stoptol*10 || maxfeas < stoptol*10)
            tmp = (LL[:trans](1.0./Lx)/n) .- 1.0
            parNCG[:count_LT] = parNCG[:count_LT] + 1
            pkkt = norm(x .- max.(x .+ tmp,0.0),2)
            if pkkt < stoptol
                stop = true
            end
        elseif (stopop == 3) && (eta < stoptol*10 || maxfeas < stoptol*10)
            tmp = (LL[:trans](1.0./Lx)/n) .- 1.0
            parNCG[:count_LT] = parNCG[:count_LT] + 1
            pkkt = norm(x .- max.(x .+ tmp,0.0),2)
            pkkt2 = maximum(tmp)
            if max(pkkt2, pkkt) < stoptol
                stop = true
            end
        elseif stopop == 4
            tmp = (LL[:trans](1.0./Lx)/n) .- 1.0
            parNCG[:count_LT] = parNCG[:count_LT] + 1
            pkkt = norm(x .- max.(x .+ tmp,0.0),2)
            if pkkt < stoptol
                stop = true
            end
        end

        ttime = time() - tstart
        runhist[:primfeas][iter] = primfeas
        runhist[:dualfeas][iter] = dualfeas
        runhist[:sigma][iter] = sigma
        runhist[:primobj][iter] = primobj
        runhist[:dualobj][iter] = dualobj
        runhist[:gap][iter] = gap
        runhist[:relgap][iter] = relgap
        runhist[:ttime][iter] = ttime
        runhist[:ttimessn][iter] = ttimessn
        runhist[:itersub][iter] = get(info_NCG, :itersub, 1) - 1
        runhist[:iterCG][iter] = get(info_NCG, :tolCG, 0)

        if printyes > 0
            @printf("\n %5d| [%3.2e %3.2e %3.2e] %- 3.2e| %- 8.6e  %- 8.6e |",
                    iter, primfeas, dualfeas, eta, relgap, primobj, dualobj)
            @printf(" %5.1f| %3.2e|", ttime, sigma)
        end

        if (stop && iter > 5) || (iter == maxiter)
            if stop
                termination = "converged"
            elseif iter == maxiter
                termination = "maxiter reached"
            end
            runhist[:termination] = termination
            runhist[:iter] = iter
            obj = [primobj, dualobj]
            break
        end

        # Update sigma
        if get(info_NCG, :breakyes, 0) >= 0
            sigma = max(sigmamin, sigma/10.0)
        elseif (iter > 1 && runhist[:dualfeas][iter]/runhist[:dualfeas][iter-1] > 0.6)
            if sigma < 1e7 && primfeas < 100*stoptol
                sigmascale = 3.0
            else
                sigmascale = sqrt(3.0)
            end
            sigma = min(sigmamax, sigma*sigmascale)
        end
    end

    info = Dict{Symbol,Any}(
        :maxfeas => maxfeas,
        :eta => eta,
        :iter => haskey(runhist, :iter) ? runhist[:iter] : maxiter,
        :relgap => relgap,
        :ttime => time() - tstart,
        :termination => termination,
        :msg => termination,
        :count_L => parNCG[:count_L],
        :count_LT => parNCG[:count_LT]
    )

    return obj, x, y, u, v, info, runhist
end
