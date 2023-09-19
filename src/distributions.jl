


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
d = Distr
t = m.R.exit[i]

"""
function lpdf(d::Weibull, t)
    # parameterization of Lee and Wang (SAS)
    #log(d.ρ) + log(d.γ) + t*(d.γ - 1.0) * log(d.ρ) - (d.ρ * t^d.γ)
    # location scale representation (Klein Moeschberger ch 12)
    # Lik: 1/sigma * exp((logt - mu)/sigma - exp((logt-mu)/sigma))
    # lLik: log(1/sigma) +  (logt - mu)/sigma - exp((logt-mu)/sigma)
    z = (log(t) - d.ρ) / d.γ
    ret = -log(d.γ) + z - exp(z)
    #ret -= log(t)   # change in variables, log transformation on t
    ret
end

function lsurv(d::Weibull, t)
    # parameterization of Lee and Wang (SAS)
    #-(d.ρ * t^d.γ)
    # location scale representation (Klein Moeschberger ch 12, modified from Wikipedia page on Gumbel Distribution)
    z = (log(t) - d.ρ) / d.γ
    ret =  -exp(z)
    #ret -= log(t)   # change in variables, log transformation on t
    ret
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

function lpdf(d::Exponential, t)
    # parameterization of Lee and Wang (SAS)
    #log(d.ρ) - (d.ρ * t)
    # location scale parameterization (Kalbfleisch and Prentice)
    log(t) - d.ρ - exp(log(t) - d.ρ)
end

function lsurv(d::Exponential, t)
    # parameterization of Lee and Wang (SAS), survival uses Kalbfleisch and Prentice
    #-d.ρ * t
    # location scale parameterization (Kalbfleisch and Prentice)
    z = log(t) - d.ρ
    ret =  -exp(z)
    ret
end

shape(d::Exponential) = 1.0
scale(d::Exponential) = d.γ
params(d::Exponential) = (d.γ)



##################
# Weibull
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

# Methods for Weibull
"""
d = Distr
t = m.R.exit[i]

"""
function lpdf(d::Lognormal, t)
    # parameterization of Lee and Wang (SAS)
    #log(d.ρ) + log(d.γ) + t*(d.γ - 1.0) * log(d.ρ) - (d.ρ * t^d.γ)
    # location scale representation (Klein Moeschberger ch 12)
    # Lik: 1/sigma * exp((logt - mu)/sigma - exp((logt-mu)/sigma))
    # lLik: log(1/sigma) +  (logt - mu)/sigma - exp((logt-mu)/sigma)
    z = (log(t) - d.ρ) / d.γ
    ret = -log(d.γ) + z - exp(z)
    #ret -= log(t)   # change in variables, log transformation on t
    ret
end

function lsurv(d::Lognormal, t)
    # parameterization of Lee and Wang (SAS)
    #-(d.ρ * t^d.γ)
    # location scale representation (Klein Moeschberger ch 12, modified from Wikipedia page on Gumbel Distribution)
    z = (log(t) - d.ρ) / d.γ
    ret =  -exp(z)
    #ret -= log(t)   # change in variables, log transformation on t
    ret
end


shape(d::Lognormal) = d.γ
scale(d::Lognormal) = d.ρ
params(d::Lognormal) = (d.ρ, d.γ)

