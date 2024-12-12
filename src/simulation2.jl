using LinearAlgebra
using Printf
using Plots


n = 1500
m = 1500
d = 2
methodtype = "ALM"
fig_option = 1
sigma_option = 1
grid_option = 1


obser, theta, SIGMA = generate_observation(n, fig_option, sigma_option, d)
grid0, mnew = select_grid(obser, grid_option, m)
L, rowmax, removeind = likelihood_matrix(obser, grid0, SIGMA, true)
if !isempty(removeind)
    n = size(L,1)
end

if methodtype == "ALM"
    options = Dict{Symbol,Any}()
    options[:scaleL] = false
    options[:approxL] = false
    options[:stoptol] = 1e-6
    options[:printyes] = true

    tstart = time()
    obj, x, y_, u_, v_, info, runhist = DualALM(L, options)
    runt = time() - tstart

    L,_,_ = likelihood_matrix(obser, grid0, SIGMA, false, false)
    llk = sum(log.(L*x))/n
    iter = info[:iter]
    @printf("iter = %d, sum(log(Lx))/n = %5.8e \n", iter, llk)
end

# Compute EB estimator and MSE
theta_hat = EB_estimator(L, x, grid0)
mse = norm(theta - theta_hat, 2)^2 / n

plot_yes = [1, 1, 1]  # [Raw+True, EB+True, G_n hat]

ms = 1
fs = 20
tiny = 0.0
xmax = 9.5; xmin = -9.5
ymax = 9.5; ymin = -9.5
x_tic = -8:2:8
y_tic = -8:2:8

if fig_option == 1
    xmin = -9.5; xmax = 9.5
    ymin = -9.5; ymax = 9.5
    x_tic = -8:2:8
    y_tic = -8:2:8
elseif fig_option == 2
    xmin = -6; xmax = 6
    ymin = -3; ymax = 9
    x_tic = -4:2:4
    y_tic = -2:2:8
elseif fig_option == 3
    xmin = -9.5; xmax = 9.5
    ymin = 3 - 9.5; ymax = 3 + 9.5
    x_tic = -8:2:8
    y_tic = -6:2:12
elseif fig_option == 4
    xmin = -9.5; xmax = 9.5
    ymin = -9.5; ymax = 9.5
    x_tic = -8:2:8
    y_tic = -8:2:8
end

xL = (xmin, xmax)
yL = (ymin, ymax)
x_txt = xL[1] + 0.03*(xL[2]-xL[1])
y_txt = yL[2] - 0.06*(yL[2]-yL[1])

# Normalize x for plotting G_n hat
x = x ./ sum(x)


nplots = sum(plot_yes)
plt = plot(layout = (1, nplots), size=(500*nplots, 500), background_color=:white)

ppp = 1

if plot_yes[1] == 1
    # Raw data and true signal
    scatter!(plt[ppp], theta[:,1], theta[:,2], color=:black, marker=(ms, :circle), label="True Signal")
    scatter!(plt[ppp], obser[:,1], obser[:,2], color=:blue, marker=(ms, :circle), label="Raw Data")
    xlims!(plt[ppp], xL)
    ylims!(plt[ppp], yL)
    xticks!(plt[ppp], x_tic)
    yticks!(plt[ppp], y_tic)
    plot!(plt[ppp], legend=:topright, legendfont=font(fs))
    ppp += 1
end

if plot_yes[2] == 1
    # EB estimator and True Signal
    scatter!(plt[ppp], theta[:,1], theta[:,2], color=:black, marker=(ms, :circle), label="True Signal")
    scatter!(plt[ppp], theta_hat[:,1], theta_hat[:,2], color=:red, marker=(ms, :circle), label="EB")
    xlims!(plt[ppp], xL)
    ylims!(plt[ppp], yL)
    xticks!(plt[ppp], x_tic)
    plot!(plt[ppp], legend=:topright, legendfont=font(fs))
    annotate!(plt[ppp], x_txt, y_txt, text("Empirical Bayes", fs))
    ppp += 1
end

if plot_yes[3] == 1
    # G_n hat
    scl = 1.0
    mx = maximum(x)
    if methodtype == "EM"
        tiny = 1e-4; mx = 1/20
    end
    for i in 1:m
        if x[i] > tiny
            scatter!(plt[ppp], [grid0[i,1]], [grid0[i,2]], color=:black, marker=(x[i]*(scl/mx), :circle), label=false)
        end
    end
    # Add true signal
    scatter!(plt[ppp], theta[:,1], theta[:,2], color=:black, marker=(ms, :circle), label="True Signal")
    xlims!(plt[ppp], xL)
    ylims!(plt[ppp], yL)
    xticks!(plt[ppp], x_tic)
    plot!(plt[ppp], ratio = :equal)
    annotate!(plt[ppp], x_txt, y_txt, text("Gₙ", fs))
    ppp += 1
end

display(plt)
