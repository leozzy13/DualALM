using LinearAlgebra
using Printf
using Arpack  
using Statistics

function DualALM(L::Matrix{Float64}, options::Dict{Symbol,Any})
    # Extract parameters
    stoptol    = get(options, :stoptol, 1e-6)
    stopop     = get(options, :stopop, 1)
    printyes   = get(options, :printyes, true)
    maxiter    = get(options, :maxiter, 100)
    sigma      = get(options, :sigma, 100.0)
    scaleL     = get(options, :scaleL, true)
    approxL    = get(options, :approxL, false)
    approxRank = get(options, :approxRank, 30)
    init_opt   = get(options, :init_opt, 0)

    if printyes
        println("\n*************************************************************************************")
        println(" ALM for the dual problem")
        println("*************************************************************************************")
    end

    tstart = time()
    n,m = size(L)

    # Scale L if needed
    if scaleL
        s = if haskey(options, :rowmax)
                1.0 ./ (options[:rowmax])
            else
                1.0 ./ (maximum(L, dims=2))
            end
        s = vec(s)
        if printyes 
            println("\n max/min scale = ", maximum(1.0./s), "/", minimum(1.0./s))
        end
        if n > 1e6
            @inbounds for i in 1:n
                L[i,:] .*= s[i]
            end
        else
            L = s .* L
        end
    else
        s = ones(n)
    end

    # Set up LL structure for times and trans operations
    LL = Dict{Symbol,Any}()
    LL[:matrix] = L

    # Low rank approximation if requested
    if approxL
        approxRank = ceil(Int, approxRank)
        approxSucceed = false
        t1 = time()
        # Perform partial SVD
        Z, nconv, niter, nmult, resid = svds(L; nsv=approxRank, ritzvec=true)
        U = Z.U  
        S = Z.S 
        V = Z.V 
        t2 = (time() - t1)
        if printyes
            println("\n approximate rank = ",approxRank)
            println(" partial svd(L) = ", t2," seconds")
            println(" singular values <= ", S[approxRank], " might be truncated")
            println(" ----------")
        end

        if S[approxRank] <= min(10*stoptol,1e-4)
            cutoff = min(10*stoptol,1e-4)
            smallidx = findfirst(x -> x < cutoff, S)
            if smallidx !== nothing
                U = U[:,1:smallidx]
                S = S[1:smallidx]
                V = V[:,1:smallidx]
                approxRank = smallidx
                if printyes 
                    println("\n approximate rank = ",approxRank," (further truncated)")
                    println(" singular values <= ", S[end], " were truncated ")
                    println(" ----------")
                end
            end
            # Incorporate S into U
            for i in 1:approxRank
                U[:,i] .*= S[i]
            end
            LL[:U] = U
            LL[:V] = V
            LL[:times] = (x-> U*(V'*x))
            LL[:trans] = (y-> V*(U'*y))
            approxSucceed = true
        end

        if !approxSucceed
            if printyes 
                println("\n numerical rank of L > ",approxRank,", set approxL = 0")
                println(" ----------")
            end
            approxL = false
            LL[:times] = (x-> L*x)
            LL[:trans] = (y-> L'*y)
        end
    else
        LL[:times] = (x-> L*x)
        LL[:trans] = (y-> L'*y)
    end

    # Initialization
    if init_opt == 0
        xnew = fill(1.0/m, m)
        ynew = vec(sum(L, dims=2)/m)
        unew = 1.0 ./ ynew
        vnew = copy(unew)
    else
        xnew = fill(0.5*sigma, m)
        ynew = vec(sum(L, dims=2)/m)
        unew = 1.0 ./ ynew
        vnew = zeros(n)
    end

    # Initial KKT check
    Lx = LL[:times](xnew)
    tmp = LL[:trans](1.0 ./ Lx)/n .- 1.0
    pkkt = norm(xnew .- max.(xnew .+ tmp, 0.0))
    if pkkt < stoptol
        # Terminated at the initial point
        obj_prim = sum(xnew) + sum(log.(s) .- log.(Lx))/n - 1
        obj = obj_prim
        y = []
        u = []
        v = []
        info = Dict{Symbol,Any}()
        runhist = Dict{Symbol,Any}()
        if printyes
            println("\n Terminated at the initial point, primal KKT residual = ", pkkt)
        end
        return obj, xnew, y, u, v, info, runhist
    end

    # Setup parmain for DualALM_main
    parmain = Dict{Symbol,Any}()
    parmain[:tstart] = tstart
    parmain[:stoptol] = stoptol
    parmain[:stopop] = stopop
    parmain[:printyes] = printyes
    parmain[:maxiter] = maxiter
    parmain[:approxL] = approxL
    parmain[:approxRank] = approxRank
    parmain[:sigma] = sigma
    parmain[:m] = m
    parmain[:n] = n

    # Call DualALM_main
    obj_main, xnew, ynew, unew, vnew, info_main, runhist_main = DualALM_main(LL, parmain, xnew, ynew, unew, vnew)
    ttime = (time() - tstart)
    iter = info_main[:iter]
    msg = info_main[:msg]
    if iter == maxiter
        msg = " maximum iteration reached"
    end

    # Rescale variables
    x = xnew
    y = ynew ./ s
    u = unew .* s
    v = vnew .* s

    Lx = L*x
    Lxorg = Lx ./ s
    Rp = Lxorg - y
    normy = norm(y)
    primfeas = max(norm(Rp)/normy, norm(min.(x,0.0))/norm(x))
    normu_val = norm(u)
    # Use vnew in Rd because original code uses 'v' before re-scaling
    Rd = max.(info_main[:count_LT] !== nothing ? LL[:trans](vnew) .- n : LL[:trans](v) .- n, 0.0)
    dualfeas = max(norm(Rd)/n, norm(u .- v)/normu_val)
    maxfeas = max(primfeas, dualfeas)
    eta = norm(y .- (1.0 ./ v))/normy

    primobj = sum(x) + sum(log.(s) .- log.(Lx))/n - 1
    dualobj = sum(log.(v))/n
    obj = [primobj, dualobj]
    gap = primobj - dualobj
    relgap = abs(gap)/(1 + abs(primobj) + abs(dualobj))

    tmp2 = ((1.0 ./ Lx')*L)'/n .- 1.0
    pkkt = norm(x .- max.(x .+ tmp2,0.0))
    pkkt2 = maximum(tmp2)

    runhist = runhist_main
    info = Dict{Symbol,Any}(
        :relgap => relgap,
        :iter => iter,
        :itersub => sum(runhist[:itersub]),
        :time => ttime,
        :timessn => sum(runhist[:ttimessn]),
        :eta => eta,
        :obj => obj,
        :maxfeas => maxfeas,
        :kktres => max(maxfeas, eta),
        :pkkt => pkkt,
        :pkkt2 => pkkt2,
        :sumlogLx => -sum(log.(Lxorg)),
        :count_L => info_main[:count_L],
        :count_LT => info_main[:count_LT],
        :termination => msg
    )

    if printyes
        println("\n****************************************")
        println(" ALM          : $msg")
        println(" iteration    : $iter")
        println(" L operator   : ", info[:count_L])
        println(" LT operator  : ", info[:count_LT])
        println(" time         : $(ttime)")
        @printf(" prim_obj     : %4.8e\n", primobj)
        @printf(" dual_obj     : %4.8e\n", dualobj)
        @printf(" relgap       : %4.5e\n", relgap)
        @printf(" primfeas     : %3.2e\n", primfeas)
        @printf(" dualfeas     : %3.2e\n", dualfeas)
        @printf(" eta          : %3.2e\n", eta)
        @printf(" primalKKT    : %3.2e\n", pkkt)
        @printf(" primalKKT2   : %3.2e\n", pkkt2)
        @printf(" -sum(log(Lx)): %1.8e\n", info[:sumlogLx])
        println(" sparsity     : ", count(x .== 0.0))
    end

    return obj, x, y, u, v, info, runhist
end

