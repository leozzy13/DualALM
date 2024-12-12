using Printf

function DualALM_main(
    LL::Dict{Symbol, Any},
    parmain,
    x::Vector{Float64},
    y::Vector{Float64},
    u::Vector{Float64},
    v::Vector{Float64}
)
    # Extract parameters with defaults
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

    # Initialize variables
    stop = false
    sigmamax = 1e7
    sigmamin = 1e-8
    count_L = 0
    count_LT = 0
    
    # Initial objective values and feasibilities
    Lx = LL[:times](x)
    count_L += 1
    obj_prim = sum(x) - sum(log.(Lx)) / n - 1.0
    obj_dual = sum(log.(u)) / n
    obj = [obj_prim, obj_dual]
    
    relgap = abs(obj_prim - obj_dual) / (1.0 + abs(obj_prim) + abs(obj_dual))
    Rp = Lx .- y
    normy = norm(y)
    primfeas = max(norm(Rp) / normy, norm(min.(x, 0.0)) / norm(x))
    LTv = LL[:trans](y)
    count_LT += 1
    Rd = max.(LTv .- n, 0.0)
    normu_val = norm(u)
    dualfeas = max(norm(Rd) / n, norm(u .- v) / max(normu_val, 1e-10))
    maxfeas = max(primfeas, dualfeas)
    eta = norm(y .- 1.0 ./ v) / normy
    
    # Initial print
    if printyes
        println("\n (dimension: m = $m, n = $n, tol = $stoptol)")
        println("---------------------------------------------------")
        println("\n*********************************************" *
                "*******************************************************")
        @printf("\n %5d| [%3.2e %3.2e %3.2e] %- 3.2e| %- 8.6e  %- 8.6e |",
                0, primfeas, dualfeas, eta, relgap, obj_prim, obj_dual)
        @printf(" %5.1f| %3.2e|\n", (time() - tstart), sigma)
    end
    
    # Prepare parameters for the main loop
    parNCG = Dict{Symbol, Any}(
        :tolconst => 0.5,
        :count_L => count_L,
        :count_LT => count_LT,
        :approxL => approxL,
        :approxRank => approxRank,
        :m => m,
        :n => n,
        :iter => 0,    # Initialize iter
        :sigma => sigma
    )
    
    maxitersub = 20
    ssncgop = Dict(
        :tol => stoptol,
        :printyes => printyes,
        :maxitersub => maxitersub  # Added maxitersub as it is used in MLE_SSNCG
    )
    
    # Prepare runhist storage
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
    
    # Main loop
    for i in 1:maxiter
        iter = i
        parNCG[:iter] = iter
        parNCG[:sigma] = sigma
    
        # Adjust maxitersub based on dualfeas
        if dualfeas < 1e-5
            maxitersub = max(maxitersub, 35)
        elseif dualfeas < 1e-3
            maxitersub = max(maxitersub, 30)
        elseif dualfeas < 1e-1
            maxitersub = max(maxitersub, 30)
        end
    
        ssncgop[:maxitersub] = maxitersub
    
        # SSN
        tstart_ssn = time()
        # Call MLE_SSNCG
        x, y, u, v, Lx, LTv, parNCG, _, info_NCG = MLE_SSNCG(LL, x, y, v, LTv, parNCG, ssncgop)
        ttimessn = time() - tstart_ssn
    
        if info_NCG[:breakyes] < 0
            parNCG[:tolconst] = max(parNCG[:tolconst] / 1.06, 1e-3)
        end
    
        # Compute KKT residual
        Rp = Lx .- y
        normy = norm(y)
        primfeas = max(norm(Rp) / normy, norm(min.(x, 0.0)) / norm(x))
        Rd = max.(LTv .- n, 0.0)
        normu_val = norm(u)
        dualfeas = max(norm(Rd) / n, norm(u .- v) / max(normu_val, 1e-10))
        maxfeas = max(primfeas, dualfeas)
        eta = norm(y .- 1.0 ./ v) / normy
    
        # Compute objective values
        primobj = sum(x) - sum(log.(Lx)) / n - 1.0
        dualobj = sum(log.(u)) / n
        obj = [primobj, dualobj]
        gap = primobj - dualobj
        relgap = abs(gap) / (1.0 + abs(primobj) + abs(dualobj))
    
        # Check stopping conditions
        if (stopop == 1) && (maxfeas < stoptol) && (eta < stoptol)
            stop = 1
        elseif (stopop == 2) && (eta < stoptol * 10 || maxfeas < stoptol * 10)
            pkkt = norm(x .- max.(x .+ (LL[:trans](1.0 ./ Lx) / n) .- 1.0, 0.0))
            parNCG[:count_LT] += 1
            if pkkt < stoptol
                stop = 1
            end
        elseif (stopop == 3) && (eta < stoptol * 10 || maxfeas < stoptol * 10)
            tmp = (LL[:trans](1.0 ./ Lx) / n) .- 1.0  # Corrected line
            parNCG[:count_LT] += 1
            pkkt = norm(x .- max.(x .+ tmp, 0.0))
            pkkt2 = maximum(tmp)
            if max(pkkt2, pkkt) < stoptol
                stop = 1
            end
        elseif stopop == 4
            tmp = (LL[:trans](1.0 ./ Lx) / n) .- 1.0  # Corrected line
            parNCG[:count_LT] += 1
            pkkt = norm(x .- max.(x .+ tmp, 0.0))
            if pkkt < stoptol
                stop = 1
            end
        end
    
        # Record run history
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
        runhist[:itersub][iter] = get(info_NCG, :itersub, 0) - 1
        runhist[:iterCG][iter] = get(info_NCG, :tolCG, 0)
    
        # Print progress if required
        if printyes
            @printf("\n %5d| [%3.2e %3.2e %3.2e] %- 3.2e| %- 8.6e  %- 8.6e |",
                    iter, primfeas, dualfeas, eta, relgap, primobj, dualobj)
            @printf(" %5.1f| %3.2e|\n", ttime, sigma)
        end
    
        # Check termination
        if (stop != 0 && iter > 5) || (iter == maxiter)
            termination = stop != 0 ? "converged" : "maxiter reached"
            runhist[:termination] = termination
            runhist[:iter] = iter
            obj = [primobj, dualobj]
            break
        end
    
        # Update sigma based on info_NCG[:breakyes] and dual feasibility
        if get(info_NCG, :breakyes, 0) >= 0  # Important to use >= 0
            sigma = max(sigmamin, sigma / 10.0)
        elseif (iter > 1 && runhist[:dualfeas][iter] / runhist[:dualfeas][iter-1] > 0.6)
            if sigma < 1e7 && primfeas < 100 * stoptol
                sigmascale = 3.0
            else
                sigmascale = sqrt(3.0)
            end
            sigma = min(sigmamax, sigma * sigmascale)
        end
    end
    
    # Prepare info dictionary
    info = Dict{Symbol, Any}(
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
