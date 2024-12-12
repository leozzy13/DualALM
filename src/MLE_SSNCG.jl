using LinearAlgebra
using Printf
using Statistics


function MLE_SSNCG(LL::Dict{Symbol, Any}, x::Vector{Float64}, y::Vector{Float64}, 
                  v::Vector{Float64}, LTv::Vector{Float64}, 
                  par::Dict{Symbol, Any}, options)
    
    # Extract options
    printyes   = options[:printyes]
    maxitersub = options[:maxitersub] 
    tol        = options[:tol]    
    
    # Initialize variables
    breakyes = 0
    maxitpsqmr = 500
    precond = 0
    stagnate_check_psqmr = 0
    sigma = par[:sigma]
    tiny = 1e-10
    n = par[:n]            
    

    v1input = v .- (n / sigma) .* y  
    
    prox_v1, M_v1, _, prox_v1_prime_m = prox_h(v1input, sigma / (n^2))
    
    v2input = (LTv / n) .+ (x / sigma) .- 1.0  
    

    prox_v2 = max.(v2input, 0.0) 
    Lprox_v2 = LL[:times](prox_v2)  
    
    par[:count_L] = get(par, :count_L, 0) + 1
    phi = -(M_v1 + (sigma / 2) * norm(prox_v2)^2)
    
    par[:precond] = precond
    par[:printyes] = printyes
    
    runhist = Dict(
        :priminf => zeros(Float64, maxitersub),
        :dualinf => zeros(Float64, maxitersub),
        :phi     => zeros(Float64, maxitersub),
        :solve_ok => zeros(Int, maxitersub),
        :psqmr    => zeros(Int, maxitersub),
        :findstep => zeros(Int, maxitersub)
    )
    
    u = zeros(Float64, length(v))
    
    itersub = 0
    for i in 1:maxitersub
        itersub = i
        
        # Compute Gradient
        tmp = (sigma / n) .* (v1input .- prox_v1)  
        Grad = (tmp .+ sigma .* Lprox_v2) ./ n    
        
        normGrad = norm(Grad)
        priminf_sub = normGrad / (norm(tmp) / n)
        
        normu = norm(prox_v1)
        dualinf_sub = max(norm(max.(LTv .- n, 0.0)) / n, norm(prox_v1 .- v) / max(normu, tiny))
        
        tolsubconst = (max(priminf_sub, dualinf_sub) < tol) ? 0.09 : 0.005
        tolsub = max(min(1e-2, get(par, :tolconst, 1.0) * dualinf_sub), tolsubconst * tol)
        
        runhist[:priminf][itersub] = priminf_sub
        runhist[:dualinf][itersub] = dualinf_sub
        runhist[:phi][itersub]     = phi
        
        if printyes
            println()
            @printf("      %5d  %- 11.10e  %3.2e %3.2e  %1.2e\n", 
                    itersub, phi, priminf_sub, dualinf_sub, get(par, :tolconst, 1.0))
        end
        
        if (priminf_sub < tolsub) && (itersub > 1)
            msg = "good termination in subproblem:"
            if printyes
                println("\n       $msg")
                @printf(" dualinf = %3.2e, normGrad = %3.2e, tolsub = %3.2e\n", 
                        dualinf_sub, priminf_sub, tolsub)
            end
            u = prox_v1
            x = sigma .* prox_v2
            Lx = sigma .* Lprox_v2
            y = (sigma / n) .* (prox_v1 .- v1input)
            breakyes = -1
            break
        end
        
        par[:epsilon] = min(1e-3, 0.1 * normGrad)
        
        if (dualinf_sub > 1e-3) || (itersub <= 5)
            maxitpsqmr = max(maxitpsqmr, 200)
        elseif (dualinf_sub > 1e-4)
            maxitpsqmr = max(maxitpsqmr, 300)
        elseif (dualinf_sub > 1e-5)
            maxitpsqmr = max(maxitpsqmr, 400)
        elseif (dualinf_sub > 5e-6)
            maxitpsqmr = max(maxitpsqmr, 500)
        end
        
        if (dualinf_sub > 1e-4)
            stagnate_check_psqmr = max(stagnate_check_psqmr, 20)
        else
            stagnate_check_psqmr = max(stagnate_check_psqmr, 30)
        end
        
        if itersub > 3 && all(runhist[:solve_ok][itersub-3:itersub-1] .<= -1) && dualinf_sub < 5e-5
            stagnate_check_psqmr = max(stagnate_check_psqmr, 80)
        end
        
        par[:stagnate_check_psqmr] = stagnate_check_psqmr
        
        if itersub > 1
            prim_ratio = priminf_sub / runhist[:priminf][itersub-1]
            dual_ratio = dualinf_sub / runhist[:dualinf][itersub-1]
        else
            prim_ratio = 0.0
            dual_ratio = 0.0
        end
        
        rhs = -Grad
        
        if get(par, :iter, 0) < 2 && itersub < 5
            tolpsqmr = min(1e-1, 0.01 * priminf_sub)
        else
            tolpsqmr = min(1e-1, 0.001 * priminf_sub)
        end

        const2 = 1.0
        if itersub > 1 && (prim_ratio > 0.5 || priminf_sub > 0.1 * runhist[:priminf][1])
            const2 *= 0.5
        end
        if dual_ratio > 1.1
            const2 *= 0.5
        end
        tolpsqmr *= const2
        
        par[:tol] = tolpsqmr
        par[:maxit] = maxitpsqmr
        par[:minitpsqmr] = 5
        
        dv, resnrm, solve_ok, par = Linsolver_MLE(rhs, LL, prox_v1_prime_m, v2input, par)
        iterpsqmr = length(resnrm) - 1  
        

        if printyes
            @printf("| %3.1e %3.1e %3d %4d %2.1f\n", 
                    par[:tol], resnrm[end], iterpsqmr, get(par, :r, 0), const2)
        end

        if ((itersub <= 3) && (dualinf_sub > 1e-4)) || (get(par, :iter, 0) <= 3)
            stepop = 1
        else
            stepop = 2
        end
        steptol = 10 * 1e-5
        
        LTdv = LL[:trans](dv)
        par[:count_LT] = get(par, :count_LT, 0) + 1
        
        phi, v1input, prox_v1, prox_v1_prime_m, v2input, prox_v2, Lprox_v2, alp, iterstep, par = 
            findstep(Grad, dv, LTdv, LL, phi, v1input, prox_v1, prox_v1_prime_m, 
                     v2input, prox_v2, Lprox_v2, steptol, stepop, par)
        
        v .= v .+ alp .* dv
        LTv .= LTv .+ alp .* LTdv
        
        runhist[:solve_ok][itersub] = solve_ok
        runhist[:psqmr][itersub]    = iterpsqmr
        runhist[:findstep][itersub] = iterstep
        
        if alp < tiny
            breakyes = 11
        end
        
        phi_ratio = 1.0
        if itersub > 1
            phi_ratio = (phi - runhist[:phi][itersub-1]) / (abs(phi) + eps())
        end
        
        if printyes
            @printf(" %3.2e %3.2e\n", alp, iterstep)
            if phi_ratio < 0
                print("-")
            end
        end
        
        if itersub > 4
            idx = max(1, itersub-3):itersub
            tmp_vals = runhist[:priminf][idx]
            ratio = minimum(tmp_vals) / maximum(tmp_vals)
            if all(runhist[:solve_ok][idx] .<= -1) && (ratio > 0.9) &&
               (minimum(runhist[:psqmr][idx]) == maximum(runhist[:psqmr][idx])) &&
               (maximum(tmp_vals) < 5 * tol)
                if printyes
                    print("#")
                end
                breakyes = 1
            end
            
            const3 = 0.7
            halfidx = ceil(Int, itersub * const3)
            priminf_1half = minimum(runhist[:priminf][1:halfidx])
            priminf_2half = minimum(runhist[:priminf][halfidx+1:itersub])
            priminf_best  = minimum(runhist[:priminf][1:itersub-1])
            priminf_ratio = runhist[:priminf][itersub] / runhist[:priminf][itersub-1]
            dualinf_ratio = runhist[:dualinf][itersub] / runhist[:dualinf][itersub-1]
            stagnate_idx = findall(x -> x <= -1, runhist[:solve_ok][1:itersub])
            stagnate_count = length(stagnate_idx)
            
            idx2 = max(1, itersub-7):itersub
            if (itersub >= 10) && all(runhist[:solve_ok][idx2] .== -1) && 
               (priminf_best < 1e-2) && (dualinf_sub < 1e-3)
                tmp2 = runhist[:priminf][idx2]
                ratio2 = minimum(tmp2) / maximum(tmp2)
                if ratio2 > 0.5
                    if printyes
                        print("##")
                    end
                    breakyes = 2
                end
            end
            
            if (itersub >= 15) && (priminf_ratio < 0.1) && (priminf_sub < 0.8 * priminf_1half) &&
               (dualinf_sub < min(1e-3, 2 * priminf_sub)) &&
               ((priminf_sub < 2e-3) || (dualinf_sub < 1e-5 && priminf_sub < 5e-3)) &&
               (stagnate_count >= 3)
                if printyes
                    print("##")
                end
                breakyes = 4
            end
            
            if (itersub >= 10) && (dualinf_sub > 5 * minimum(runhist[:dualinf])) && 
               (priminf_sub > 2 * minimum(runhist[:priminf]))
                if printyes
                    print("###")
                end
                breakyes = 5
            end
            
            if (itersub >= 20)
                dualinf_ratioall = runhist[:dualinf][2:itersub] ./ runhist[:dualinf][1:itersub-1]
                idx_increase = findall(x -> x > 1, dualinf_ratioall)
                if length(idx_increase) >= 3
                    dualinf_increment = mean(dualinf_ratioall[idx_increase])
                    if dualinf_increment > 1.25
                        if printyes
                            print("^^")
                        end
                        breakyes = 6
                    end
                end
            end
        end
        
        if breakyes != 0
            break
        end
    end
    
    itersub = min(itersub, maxitersub)

    if itersub == maxitersub
        u = prox_v1
        x = sigma .* prox_v2
        Lx = sigma .* Lprox_v2
        y = (sigma / n) .* (prox_v1 .- v1input)
    else
        Lx = sigma .* Lprox_v2
    end
    
    info = Dict{Symbol, Any}()
    info[:tolCG] = sum(runhist[:psqmr][1:itersub])
    info[:breakyes] = breakyes
    info[:itersub] = itersub
    
    return x, y, u, v, Lx, LTv, par, runhist, info
end
