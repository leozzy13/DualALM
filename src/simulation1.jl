using Random, Distributions, Statistics, StatsFuns, Plots, LinearAlgebra, Arpack, SparseArrays

n = 1000
m = 500
k = 500
mu = 10

obser = zeros(Float64, n)
obser[1:k] .= randn(k) .+ mu
obser[k+1:end] .= randn(n - k)

ub = maximum(obser) + eps(Float64)
lb = minimum(obser) - eps(Float64)
grid = collect(range(lb, ub, length=m)) 
diffM = obser .- grid'  
L = pdf.(Normal(0, 1), diffM) 

options = Dict{Symbol, Any}(
    :maxiter => 100,
    :stoptol => 1e-6,
    :stopop => 3,
    :printyes => true,
    :approxL => true,
    :init_opt => 0
)

obj, x, y, u, v, info, runhist = DualALM(L, options)

obsersort = sort(obser)
id = sortperm(obser)

xx = collect(-maximum(grid) - 1:0.01:maximum(grid) + 1)
yy = (k / n) * pdf.(Normal(mu, 1), xx) .+ (n - k) / n * pdf.(Normal(0, 1), xx)
theta_hat = (L * (grid .* x)) ./ (L * x)

default(
    legend = :topright,
    xlabel = "x",
    ylabel = "Density",
    linewidth = 2,
    markersize = 4,
    guidefontsize = 15,        
    tickfontsize = 12,       
    titlefontsize = 16,      
    legendfontsize = 15
)

p1 = plot(xx, yy, label = "True density \$f_{G^*,1}\$", lw = 2)
plot!(p1, obsersort, y[id], linestyle = :dot, label = "Estimated mixture density \$widehat{f}_{widehat{G}_n,1}\$")

p2 = plot(grid, x, label = "NPMLE of the prior probability measure \$widehat{G}_n\$", lw = 2)

p3 = plot(obsersort, theta_hat[id], marker = :+, label = "Bayes estimator of \$\theta_i\$", lw = 2)

plot(p1, p2, p3, layout = (1, 3), size = (2200, 1000))
