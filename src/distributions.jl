


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
