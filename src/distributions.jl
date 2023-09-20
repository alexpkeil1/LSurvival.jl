


######################################################################
# lightweight replacement functions for some distributions used in 
# intermediate calculations
######################################################################

"""
quantile function for a standard normal distribution
    depends on SpecialFunctions
    https://en.wikipedia.org/wiki/Normal_distribution
"""
qstdnorm(p) = sqrt(2) * erfinv(2.0 * p - 1.0)


"""
quantile function for a standard normal distribution
    depends on SpecialFunctions
    https://en.wikipedia.org/wiki/Normal_distribution
"""
cdfnorm(z) = 0.5 * (1 + erf(z/sqrt(2)))


"""
quantile function for a chi-squared distribution
    depends on SpecialFunctions
    https://en.wikipedia.org/wiki/Chi-squared_distribution
"""
cdfchisq(df, x) = gamma_inc(df / 2, x / 2, 0)[1]


"""
p-value for a (null) standard normal distribution
    depends on SpecialFunctions
    https://en.wikipedia.org/wiki/Normal_distribution
"""
calcp(z) = 1.0 - erf(abs(z) / sqrt(2))


"""
Quantile function for the Weibull distribution

    lightweight function used for simulation

        Note that there is no checking that parameters ρ,γ are positively bound, and p ∈ (0,1), and errors will be given if this is not the case

Signature:

```julia        
qweibull(p::Real,ρ::Real,γ::Real)
```
# quantile(Weibull(.75, 1.1), .3) # cross reference the approach in the Distributions package
"""
qweibull(p, ρ, γ) = γ * ((-log1p(-p))^(1 / ρ))


"""
Random draw from Weibull distribution

lightweight function used for simulation

Note that there is no checking that parameters ρ,γ are positively bound, and errors will be given if this is not the case

Signatures:

```julia
randweibull(rng::MersenneTwister,ρ::Real,γ::Real)
randweibull(ρ::Real,γ::Real)
```
"""
randweibull(rng, ρ, γ) = qweibull(rand(rng), ρ, γ)
randweibull(ρ, γ) = randweibull(MersenneTwister(), ρ, γ)

######################################################################
# Distributions used in parametric survival models
######################################################################


##################
# Weibull
##################
struct Weibull{T<:Real} <: AbstractSurvDist
    ρ::T   # scale: linear effects on this parameter
    γ::T   # shape
end

function Weibull(ρ::T, γ::T) where {T<:Int}
    Weibull(Float64(ρ), Float64(γ))
end

function Weibull(ρ::T, γ::R) where {T<:Int,R<:Float64}
    Weibull(Float64(ρ), γ)
end

function Weibull(ρ::R, γ::T) where {R<:Float64,T<:Int}
    Weibull(ρ, Float64(γ))
end

function Weibull(v::Vector{R}) where {R<:Real}
    Weibull(v[1], v[2])
end

function Weibull()
    Weibull(ones(Float64, 2)...)
end

# Methods for Weibull
"""
log probability distribution function: Weibull distribution

# location scale representation (Klein Moeschberger ch 12)
# Lik: λγ(λt)^{γ-1}exp(-(λt)^γ) # traditional Weibull
# Lik: exp(-α)(1/σ) (exp(-α)exp(log(t)))^{(1/σ)-1} exp(-(exp(-α)exp(log(t)))^(1/σ))           , γ=σ^-1, λ=exp(-α), t = exp(log(t))
# Lik: (1/σ)exp(-α)  exp(log(t)-α)^{(1/σ)-1} exp(-(exp(log(t)-α)/σ))           
# Lik: 1/σ *exp(-α) exp((log(t)-α)/σ-(log(t)-α)) exp(-(exp(log(t)-α)/σ))           
# lLik: log(1/σ) - α +  (log(t)-α)/σ -log(t)+α       - exp(log(t)-α)/σ)           
# lLik: log(1/σ)     +  z            -log(t)         - exp(z)                 z = (log(t)-α)/σ
# lLik: -log(σ) + z - exp(z) - log(t),                                        σ=d.γ=γ^-1, α=d.ρ=-log(λ)

"""
function lpdf_weibull(ρ, γ, t)
    z = (log(t) - ρ) / γ
    ret = -log(γ) + z - exp(z) -log(t)
    ret
end

function lpdf(d::Weibull, t)
    lpdf_weibull(d.ρ, d.γ, t)
end

"""
# location scale representation (Klein Moeschberger ch 12)
# Surv: exp(-(λt)^γ) # traditional Weibull
# Surv: exp(-(exp(-α)exp(log(t)))^(1/σ)) , γ=σ^-1, λ=exp(-α), t = exp(log(t))
# Surv: exp(-exp((log(t)-α)/σ)) 
# lSurv: -exp((log(t)-α)/σ)
# lSurv: -exp(z)

"""
function lsurv_weibull(ρ, γ, t)
    z = (log(t) - ρ) / γ
    ret = -exp(z)
    ret
end

function lsurv(d::Weibull, t)
    # location scale representation (Klein Moeschberger ch 12, modified from Wikipedia page on Gumbel Distribution)
    lsurv_weibull(d.ρ, d.γ, t)
end


shape(d::Weibull) = d.γ
scale(d::Weibull) = d.ρ
params(d::Weibull) = (d.ρ, d.γ)



##################
# Exponential
##################
mutable struct Exponential{T<:Real} <: AbstractSurvDist
    ρ::T   # scale (Weibull shape is 1.0)
end

function Exponential(ρ::T) where {T<:Int}
    Exponential(Float64(ρ))
end

function Exponential(v::Vector{R}) where {R<:Real}
    length(v) > 1 &&
        throw("Vector of arguments given to `Exponential()`; did you mean Exponential.() ?")
    Exponential(v[1])
end

function Exponential()
    Exponential(one(Float64))
end

# Methods for exponential

"""
Derivation in lsurv(d::Weibull), setting σ=d.γ=1
"""
function lpdf(d::Exponential, t)
    # location scale parameterization (Kalbfleisch and Prentice)
    - d.ρ - t*exp(-d.ρ) 
end
"""
Derivation in lsurv(d::Weibull), setting σ=d.γ=1
"""
function lsurv(d::Exponential, t)
    # location scale parameterization (Kalbfleisch and Prentice)
    z = log(t) - d.ρ
    ret =  -exp(z)
    ret
end

shape(d::Exponential) = 1.0
scale(d::Exponential) = d.γ
params(d::Exponential) = (d.γ)



##################
# Lognormal
##################
struct Lognormal{T<:Real} <: AbstractSurvDist
    ρ::T   # scale: linear effects on this parameter
    γ::T   # shape
end

function Lognormal(ρ::T, γ::T) where {T<:Int}
    Lognormal(Float64(ρ), Float64(γ))
end

function Lognormal(ρ::T, γ::R) where {T<:Int,R<:Float64}
    Lognormal(Float64(ρ), γ)
end

function Lognormal(ρ::R, γ::T) where {R<:Float64,T<:Int}
    Lognormal(ρ, Float64(γ))
end

function Lognormal(v::Vector{R}) where {R<:Real}
    Lognormal(v[1], v[2])
end

function Lognormal()
    Lognormal(ones(Float64, 2)...)
end

# Methods for Lognormal
"""
log probability distribution function: Weibull distribution

# location scale representation (Klein Moeschberger ch 12)
# Lik: (2π)^(-1/2)γ/t exp((-γ^2 log(λt)^2)/2)       # traditional Lognormal
# Lik: (2π)^(-1/2)(1/σ)/exp(log(t)) exp((-(1/σ)^2 log(exp(-α)exp(log(t)))^2)/2)           , γ=σ^-1, λ=exp(-α), t = exp(log(t))
# Lik: (2π)^(-1/2)(1/σt) exp((-(1/σ)^2 log(exp(log(t)-α))^2)/2)
# Lik: (2π)^(-1/2)(1/σt) exp((-1/2 * (log(t)-α)/σ)^2))
# Lik: (2π)^(-1/2)(1/σt) exp((-1/2 * z^2))
# lLik: log(-sqrt(2π σ^2 t^2)) -z^2/2                                        σ=d.γ=γ^-1, α=d.ρ=-log(λ)

"""
function lpdf(d::Lognormal, t)
    # location scale representation (Klein Moeschberger ch 12)
    z = (log(t) - d.ρ) / d.γ
    ret = log(inv(sqrt(2pi)*d.γ*t)) 
    ret += -z*z/2.0
    ret
end

"""
log probability distribution function: Weibull distribution

# location scale representation (Klein Moeschberger ch 12)
# Surv: 1- Φ(γlog(λt))      # traditional Lognormal
# Surv: 1- Φ((1/σ)log(exp(-α)exp(log(t))))                        , γ=(1/σ), λ=exp(-α), t = exp(log(t))
# Surv: 1- Φ((log(t)-α)/σ)
# iSurv: log(1-Φ(z))                                        σ=d.γ=γ^-1, α=d.ρ=-log(λ)
"""
function lsurv(d::Lognormal, t)
    # location scale representation (Klein Moeschberger ch 12, modified from Wikipedia page 
    z = (log(t) - d.ρ) / d.γ
    ret =  log(1-cdfnorm(z))
    ret
end


shape(d::Lognormal) = d.γ
scale(d::Lognormal) = d.ρ
params(d::Lognormal) = (d.ρ, d.γ)

