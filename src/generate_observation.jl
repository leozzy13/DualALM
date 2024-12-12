# This function generates observations in R^2 based on different options for distributions.
# Inputs:
#   n: Number of observations
#   fig_option: Distribution type to generate (e.g., concentric circles, triangle, digit 8, etc.)
#   sigma_option: Type of covariance structure for noise
#   d: Dimension of data (default is 2)
# Outputs:
#   X: Observations (n rows)
#   theta: True signal (n rows)
#   SIGMA: Covariance matrix for each observation if sigma_option is 2


function generate_observation(n, fig_option, sigma_option = 1, d = 2)
    if d < 2
        d = 2
    end

    if sigma_option == 1
        SIGMA = Matrix{Float64}(I, d, d)
    elseif sigma_option == 2
        SIGMA = rand(Uniform(1, 3), 1, d, n)
    else
        error("Invalid sigma_option. It must be either 1 or 2.")
    end

    theta = zeros(n, d)
    X = zeros(n, d)

    if fig_option == 1
        # Two concentric circles
        r1, r2 = 2, 6
        n1 = round(Int, n / 2)
        n2 = n - n1
        t = 2 * π * rand(n1)
        theta[1:n1, :] .= r1 .* hcat(cos.(t), sin.(t))
        t = 2 * π * rand(n2)
        theta[n1+1:n, :] .= r2 .* hcat(cos.(t), sin.(t))

    elseif fig_option == 2
        # Triangle
        n1 = round(Int, n / 3)
        n2 = n1
        n3 = n - n1 - n2
        p1 = [-3, 0] 
        p2 = [0, 6]
        p3 = [3, 0]

        theta[1:n1, :] .= hcat([p1 .+ (p2 - p1) * a for a in rand(n1, 1)]...)'
        theta[n1+1:2*n1, :] .= hcat([p2 .+ (p3 - p2) * a for a in rand(n2, 1)]...)'
        theta[2*n1+1:end, :] .= hcat([p3 .+ (p1 - p3) * a for a in rand(n3, 1)]...)'

    elseif fig_option == 3
        # Digit 8 (two circles)
        r1, c1 = 3, [0, 0]
        r2, c2 = 3, [0, 6]
        n1 = round(Int, n / 2)
        n2 = n - n1
        t = 2 * π * rand(n1)
        theta[1:n1, :] .= hcat([c1 .+ r1 * [cos.(a), sin.(a)] for a in t]...)'
        t = 2 * π * rand(n2)
        theta[n1+1:n, :] .= hcat([c2 .+ r2 * [cos.(a), sin.(a)] for a in t]...)'

    elseif fig_option == 4
        # Letter A (five segments)
        n1 = round(Int, n / 5)
        n2, n3, n4, n5 = n1, n1, n1, n - (4 * n1)
        p1, p2, p3, p4, p5 = [-4, -6], [-2, 0], [0, 6], [2, 0], [4, -6]
        theta[1:n1, :] .= hcat([p1 .+ (p2 - p1) * a for a in rand(n1, 1)]...)'
        theta[n1+1:2*n1, :] .= hcat([p2 .+ (p3 - p2) * a for a in rand(n2, 1)]...)'
        theta[2*n1+1:3*n1, :] .= hcat([p3 .+ (p4 - p3) * a for a in rand(n3, 1)]...)'
        theta[3*n1+1:4*n1, :] .= hcat([p4 .+ (p5 - p4) * a for a in rand(n4, 1)]...)'
        theta[4*n1+1:n, :] .= hcat([p5 .+ (p2 - p4) * a for a in rand(n5, 1)]...)'

    elseif fig_option == 5
        # Circle of radius 6
        r = 6
        t = 2 * π * rand(n)
        theta[:, 1:2] .= hcat([r * [cos(a), sin(a)] for a in t]...)'

    elseif fig_option == 6
        # theta_i = 0
        theta .= 0

    elseif fig_option == 7
        # theta_i = 0, 6 * e1, or 6 * e2
        r = 6
        atoms = zeros(3, d)
        atoms[2, 1] = r
        atoms[3, 2] = r
        xstar = ones(3) / 3 
        xstar2 = cumsum(xstar)
        kk = 0
        for i in 1:3
            kend = round(Int, n * xstar2[i])
            theta[(kk+1):kend, :] .= repeat(reshape(atoms[i, :], 1, d), kend - kk, 1)
            kk = kend
        end

    elseif fig_option == 8
        # theta_i ~ N(0, SIGMA)
        if sigma_option == 1
            theta = rand(MvNormal(zeros(d), SIGMA), n)'
        else
            for i in 1:n
                theta[i, :] = rand(MvNormal(zeros(d), Diagonal(vec(SIGMA[:, :, i]))))
            end
        end

    elseif fig_option == 9
        r = 6
        atoms = zeros(6, d)
        atoms[2, 1] = r
        atoms[3, 1] = -r
        atoms[4, 2] = r
        atoms[5, 1:2] = [r, r]
        atoms[6, 1:2] = [-r, r]
        xstar = ones(6) / 6  
        xstar2 = cumsum(xstar)
        kk = 0
        for i in 1:6
            kend = round(Int, n * xstar2[i])
            theta[(kk+1):kend, :] .= repeat(reshape(atoms[i, :], 1, d), kend - kk, 1)
            kk = kend
        end

    else
        error("Invalid fig_option. Must be an integer from 1 to 9.")
    end

    for i in 1:n
        if sigma_option == 1
            X[i, :] = rand(MvNormal(theta[i, :], SIGMA))
        else
            X[i, :] = rand(MvNormal(theta[i, :], Diagonal(vec(SIGMA[:, :, i]))))
        end
    end

    return X, theta, SIGMA
end

# function test(complete in all cases)
n = 5
fig_option = 8
sigma_option = 2
d = 2
X, theta, SIGMA = generate_observation(n, fig_option, sigma_option, d)
println("X: ", X)
println("Theta: ", theta)
println("SIGMA: ", SIGMA)