# Inputs:
#   L: Matrix of size (n, m) - corresponds to L \in R^{n*m}
#   x: Vector of size (m,) - corresponds to x \in R^m, x >= 0, sum xi = 1
#   U: Matrix of size (m, d) - corresponds to U \in R^{m*d}
# Outputs:
#   theta_hat: Matrix of size (n, d) - corresponds to theta_hat \in R^{n*d}
function EB_estimator(L::Matrix{Float64}, x::Vector{Float64}, U::Matrix{Float64})

    Lx = L * x
    n, _ = size(L)
    d = size(U, 2)
    theta_hat = zeros(n, d)

    for i in 1:n
        # Multiply the i-th row of L element-wise with U, then take dot product with x
        theta_hat[i, :] = (x' * (L[i, :] .* U)) / Lx[i]
    end
    return theta_hat
end



## function test(complete)
L = [11.0 2.0 3.0; 4.0 55.0 6.0]
x = [0.2, 0.3, 0.5]
U = [0.1 0.2; 0.3 0.4; 0.5 0.6]
theta_hat = EB_estimator(L, x, U)

println("theta_hat:")
println(theta_hat)