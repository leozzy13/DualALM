# Function to compute the proximal mapping of h(y) = (-1/n) * sum(log(y_j)), y ∈ R^n 
# Arguments:
#   y            : Input vector of length n.
#   sigma        : Scalar parameter.
#
# Returns:
#   prox_y       : Proximal point of y.
#   M_y          : Value of the Moreau envelope at y.
#   prox_prime   : Diagonal vector of the derivative of the proximal mapping.
#   prox_prime_minus : 1 - prox_prime.


using LinearAlgebra

function prox_h(y, sigma)
    n = length(y)
    tmp = sqrt.(y.^2 .+ 4.0 / (sigma * n))
    prox_y = 0.5 .* (tmp .+ y)
    
    if any(prox_y .<= 0)
        @warn "log is not defined for non-positive elements, shifting prox_y"
        prox_y .= prox_y .+ 1e-30
    end
    
    M_y = (sigma / 2.0) * norm(prox_y - y)^2 - (sum(log.(prox_y)) / n)
    
    tmp = y ./ tmp
    prox_prime = 0.5 .* (1 .+ tmp)
    prox_prime_minus = 0.5 .* (1 .- tmp)
    
    return prox_y, M_y, prox_prime, prox_prime_minus
end




## function test(complete)
y = [0.5, 1.0, -0.3, 0.8, -0.1]
sigma = 0.1

prox_y, M_y, prox_prime, prox_prime_minus = prox_h(y, sigma)

println("prox_y: ", prox_y)
println("M_y: ", M_y)
println("prox_prime: ", prox_prime)
println("prox_prime_minus: ", prox_prime_minus)