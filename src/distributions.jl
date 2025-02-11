


######################################################################
# lightweight replacement functions for some distributions used in 
# intermediate calculations
######################################################################


"""
quantile function for a standard normal distribution
    depends on SpecialFunctions
    https://en.wikipedia.org/wiki/Normal_distribution

    Source code, example:

```julia
    qstdnorm(p) = sqrt(2) * SpecialFunctions.erfinv(2.0 * p - 1.0)
    
    qstdnorm(.975)
```

"""
qstdnorm(p) = sqrt(2) * erfinv(2.0 * p - 1.0)


"""
quantile function for a standard normal distribution
    depends on SpecialFunctions
    https://en.wikipedia.org/wiki/Normal_distribution
    
Source code, example:

```julia
    cdfnorm(z) = 0.5 * (1 + SpecialFunctions.erf(z / sqrt(2)))
    
    cdfnorm(1.96)
```
    
"""
cdfnorm(z) = 0.5 * (1 + erf(z / sqrt(2)))


"""
quantile function for a chi-squared distribution
    depends on SpecialFunctions
    https://en.wikipedia.org/wiki/Chi-squared_distribution

Source code, example:

```julia
    cdfchisq(df, x) = SpecialFunctions.gamma_inc(df / 2, x / 2, 0)[1]

    cdfchisq(3, 3.45)
```

"""
cdfchisq(df, x) = gamma_inc(df / 2, x / 2, 0)[1]


"""
Two-tailed p-value for a (null) standard normal distribution
    depends on SpecialFunctions
    https://en.wikipedia.org/wiki/Normal_distribution

```julia
    calcp(z) = 1.0 - SpecialFunctions.erf(abs(z) / sqrt(2))

    calcp(1.96)
```

    """
calcp(z) = 1.0 - erf(abs(z) / sqrt(2))


"""
Quantile function for the Weibull distribution
    
```math
F(t) = \begin{cases}
     1 - e^{{-t/ρ}^{α}}&, t ≥ 0\\
     0&, t < 0
\end{cases}
Q(p) = ρ * (log(1/(1-p))^{1/α})
```

lightweight function used for simulation

Note that there is no checking that parameters α,ρ are positively bound, and p ∈ (0,1), and errors will be given if this is not the case

Signature:

```julia        
qweibull(p::Real,α::Real,ρ::Real)
```

Source code, example:

```julia
qweibull(p, α, ρ) = ρ * ((-log1p(-p))^(1 / α))

# cross reference the approach in the Distributions package
quantile(Distributions.Weibull(.75, 1.1), .3)
LSurvival.qweibull(0.3, .75, 1.1)
```
"""
qweibull(p, α, ρ) = ρ * ((-log1p(-p))^(1 / α))


"""
Random draw from Weibull distribution

lightweight function used for simulation

Note that there is no checking that parameters α,ρ are positively bound, and errors will be given if this is not the case

Signatures:

```julia
randweibull(rng::MersenneTwister,α::Real,ρ::Real)
randweibull(α::Real,ρ::Real)
```

Source code, example:

```julia
randweibull(rng, α, ρ) = qweibull(rand(rng), α, ρ)
randweibull(α, ρ) = randweibull(MersenneTwister(), α, ρ)

# cross reference the approach in the Distributions package
rand(Distributions.Weibull(.75, 1.1))
randweibull(.75, 1.1)
```


"""
randweibull(rng, α, ρ) = qweibull(rand(rng), α, ρ)
randweibull(α, ρ) = randweibull(MersenneTwister(), α, ρ)




######################################################################
# Common functions for all survival distributions
######################################################################

location(d::D) where {D<:AbstractSurvDist} = shape(d)


######################################################################
# Distributions used in parametric survival models
######################################################################


raw"""
Mean model parameter gradient function for linear model
    $\alpha = f(\theta, X) = \theta X$
    
dαdβ returns a vector of partial derivatives of $\alpha$ with respect to the vector $\theta$
"""
function dαdβ(β, X)
    [x for x in X]
end



##############################
# Weibull distribution 
##############################
struct Weibull{T<:Real} <: AbstractSurvDist
    α::T   # shape: linear effects on this parameter
    ρ::T   # scale
end

function Weibull(α::T, ρ::T) where {T<:Int}
    Weibull(Float64(α), Float64(ρ))
end

function Weibull(α::T, ρ::R) where {T<:Int,R<:Float64}
    Weibull(Float64(α), ρ)
end

function Weibull(α::R, ρ::T) where {R<:Float64,T<:Int}
    Weibull(α, Float64(ρ))
end

function Weibull()
    Weibull(ones(Float64, 2)...)
end

# Methods for Weibull
"""
Log probability distribution function: Weibull distribution

# location scale representation (Klein Moeschberger ch 12)

```julia
α=0.1   # location
ρ=-1.2  # log(scale)
time=2
lpdf(Weibull(α, ρ), time)
```
"""
function lpdf(d::Weibull, t)
    lpdf_weibull(d.α, d.ρ, t)
end

"""
Log survival distribution function: Weibull distribution
    
# location, log(scale) representation (Klein Moeschberger ch 12)

```julia
α=0.1   # location
ρ=-1.2  # log(scale)
time=2
lsurv(Weibull(α, ρ), time)
```

"""
function lsurv(d::Weibull, t)
    # location scale representation (Klein Moeschberger ch 12, modified from Wikipedia page on Gumbel Distribution)
    lsurv_weibull(d.α, d.ρ, t)
end


"""
Log-likelihood calculation for weibull regression: PDF

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Weibull()
lpdf(d, vcat(θ,ρ), t, x)
```
"""
function lpdf(d::Weibull, θ, t, x)
    lpdf_weibull(dot(θ[1:end-1], x), θ[end], t)
end

"""
Log-likelihood calculation for weibull regression: Survival

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Weibull()
lsurv(d, vcat(θ,ρ), t, x)
```
"""
function lsurv(d::Weibull, θ, t, x)
    lsurv_weibull(dot(θ[1:end-1], x), θ[end], t)
end

"""
Gradient calculation for weibull regression: PDF

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
#α = dot(β,x)
d = Weibull()
lpdf_gradient(d, vcat(θ,ρ), t, x)
```
"""
function lpdf_gradient(d::Weibull, θ, t, x)
    dlpdf_regweibull(θ[1:end-1], θ[end], t, x)
end

"""
Gradient calculation for weibull regression: Survival

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Weibull()
lsurv_gradient(d, vcat(θ,ρ), t, x)
```
"""
function lsurv_gradient(d::Weibull, θ, t, x)
    dlsurv_regweibull(θ[1:end-1], θ[end], t, x)
end


"""
Hessian calculation for weibull regression: PDF

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Weibull()
lpdf_hessian(d, vcat(θ,ρ), t, x)
```
"""
function lpdf_hessian(d::Weibull, θ, t, x)
    ddlpdf_regweibull(θ[1:end-1], θ[end], t, x)
end

"""
Hessian calculation for weibull regression: Survival
```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Weibull()
lsurv_hessian(d, vcat(θ,ρ), t, x)
```

"""
function lsurv_hessian(d::Weibull, θ, t, x)
    ddlsurv_regweibull(θ[1:end-1], θ[end], t, x)
end

raw"""
Hessian calculation for Weibull distribution: PDF

    $$f(t;\rho,\beta) = \exp(\gamma)^{-1}\exp(\ln(t)-\rho)/\exp(\gamma))\exp(-\exp((\ln(t)-\rho)/\exp(\gamma)))$$
    $$\ln\mathfrak{L}(t;\rho,\beta) = -2\gamma + \ln(t)-\rho  -\exp((\ln(t)-\rho)\exp(-\gamma))$$

```
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Weibull(α, ρ)
lpdf_hessian(d, t)
```
"""
function lpdf_hessian(d::Weibull, t)
    ddlpdf_weibull(d.α, d.ρ, t)
end

"""
Hessian calculation for Weibull distribution: Survival

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Weibull(α, ρ)
lsurv_hessian(d, t)
```
"""
function lsurv_hessian(d::Weibull, t)
    ddlsurv_weibull(d.α, d.ρ, t)
end


#scalelocation(d::Weibull) = d.ρ
#location(d::Weibull) = d.α
logscale(d::Weibull) = d.ρ
scale(d::Weibull) = exp(logscale(d))
shape(d::Weibull) = d.α
params(d::Weibull) = (d.α, d.ρ)

################################################
# underlying distribution functions, Weibull distribution
################################################
#=
# pg 34 of Kalbfleisch and Prentice
$$f(t;\rho,\beta) = \exp(\gamma)^{-1}\exp(\ln(t)-\rho)/\exp(\gamma))\exp(-\exp((\ln(t)-\rho)/\exp(\gamma)))$$
$$\ln\mathfrak{L}(t;\rho,\beta) = -2\gamma + \ln(t)-\rho  -\exp((\ln(t)-\rho)\exp(-\gamma))$$

=#
"""
α = -1.2
ρ = 1.8
t = 4.3
z = (log(t) - α) * exp(-ρ)
z - exp(z) - ρ - log(t)

"""
function lpdf_weibull(α, ρ, t)
    # l = -α*exp(-ρ) - exp(- α*exp(-ρ) + log(t)*exp(-ρ)) + log(t)*exp(-ρ) - ρ - log(t)
    z = (log(t) - α) * exp(-ρ)
    z - exp(z) - ρ - log(t)
end
function lsurv_weibull(α, ρ, t)
    z = (log(t) - α) * exp(-ρ)
    -exp(z)
end

lpdf_weibull(α::Int, ρ::Float64, t) = lpdf_weibull(Float64(α), ρ::Float64, t)
lpdf_weibull(α::Float64, ρ::Int, t) = lpdf_weibull(α::Float64, Float64(ρ), t)
lpdf_weibull(α::Int, ρ::Int, t) = lpdf_weibull(Float64(α), Float64(ρ), t)
lsurv_weibull(α::Int, ρ::Float64, t) = lsurv_weibull(Float64(α), ρ::Float64, t)
lsurv_weibull(α::Float64, ρ::Int, t) = lsurv_weibull(α::Float64, Float64(ρ), t)
lsurv_weibull(α::Int, ρ::Int, t) = lsurv_weibull(Float64(α), Float64(ρ), t)



################################################
# underlying gradients, Weibull distribution
################################################
#=
α = -1.2
ρ = 1.8
t = 4.3
z = (log(t) - α) * exp(-ρ)

exp(z-ρ) - exp(-ρ)
# wolfram alpha
exp(-ρ) * (exp(α*(-exp(-ρ))) * t^(exp(-ρ)) - 1)

expm1(z)*z - 1
# wolfram alpha
α*exp(-ρ) - exp(z)*(-z) - exp(-ρ)*log(t) - 1
=#

function dlpdf_weibull(α, ρ, t)
    z = (log(t) - α) * exp(-ρ)
    [exp(z-ρ) - exp(-ρ), expm1(z)*z - 1]
end

function dlsurv_weibull(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    [exp(z-ρ), z * exp(z)]
end

dlpdf_weibull(α::Int, ρ::Float64, t) = dlpdf_weibull(Float64(α), ρ::Float64, t)
dlpdf_weibull(α::Float64, ρ::Int, t) = dlpdf_weibull(α::Float64, Float64(ρ), t)
dlpdf_weibull(α::Int, ρ::Int, t) = dlpdf_weibull(Float64(α), Float64(ρ), t)
dlsurv_weibull(α::Int, ρ::Float64, t) = dlsurv_weibull(Float64(α), ρ::Float64, t)
dlsurv_weibull(α::Float64, ρ::Int, t) = dlsurv_weibull(α::Float64, Float64(ρ), t)
dlsurv_weibull(α::Int, ρ::Int, t) = dlsurv_weibull(Float64(α), Float64(ρ), t)


################################################
# underlying Hessians, Weibull distribution
################################################

function ddlpdf_weibull(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    vcat(
        hcat(-exp(z-2ρ), exp(-ρ) - (z + 1.0) * exp(z - ρ)),
        hcat(exp(-ρ) - (z + 1.0) * exp(z - ρ), z-(z+1)*exp(z)*z),
    )
end

function ddlsurv_weibull(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    vcat(
        hcat(-exp(z-2ρ), -(z + 1.0) * exp(z - ρ)),
        hcat(-(z + 1.0) * exp(z - ρ), -z*(1.0+z) * exp(z)),
    )
end

ddlpdf_weibull(α::Int, ρ::Float64, t) = ddlpdf_weibull(Float64(α), ρ::Float64, t)
ddlpdf_weibull(α::Float64, ρ::Int, t) = ddlpdf_weibull(α::Float64, Float64(ρ), t)
ddlpdf_weibull(α::Int, ρ::Int, t) = ddlpdf_weibull(Float64(α), Float64(ρ), t)
ddlsurv_weibull(α::Int, ρ::Float64, t) = ddlsurv_weibull(Float64(α), ρ::Float64, t)
ddlsurv_weibull(α::Float64, ρ::Int, t) = ddlsurv_weibull(α::Float64, Float64(ρ), t)
ddlsurv_weibull(α::Int, ρ::Int, t) = ddlsurv_weibull(Float64(α), Float64(ρ), t)


################################################
# Underlying gradients, Weibull regression
################################################

function dlsurv_regweibull(β, ρ, t, x)
    dα = dαdβ(β, x)
    df = dlsurv_weibull(dot(β, x), ρ, t)
    dfdβ = [dα[j] * df[1] for j = 1:length(β)]
    dsdβ = vcat(dfdβ, df[2])
    dsdβ
end

function dlpdf_regweibull(β, ρ, t, x)
    dα = dαdβ(β, x)
    df = dlpdf_weibull(dot(β, x), ρ, t)
    dfdβ = [dα[j] * df[1] for j = 1:length(β)]
    dfdβ = vcat(dfdβ, df[2])
    dfdβ
end


################################################
# Underlying Hessians, Weibull regression
################################################

function ddlpdf_regweibull(β, ρ, t, x)
    dα = dαdβ(β, x)
    ddf = ddlpdf_weibull(dot(β, x), ρ, t)
    #dfdβ = [dα[j] * df[1] for j in length(β)]   
    nb = length(β)
    np = nb + length(ρ)
    ddsdβ = zeros(np, np)
    ddsdβ[1:nb, np:np] .= ddf[1, 2] .* dα
    ddsdβ[np:np, 1:nb] .= ddf[2, 1] .* dα'
    ddsdβ[np, np] = ddf[2, 2]
    for r = 1:nb
        for c = r:nb
            ddsdβ[r, c] = ddsdβ[c, r] = ddf[1, 1] * dα[r] * dα[c]
        end
    end
    ddsdβ
end

function ddlsurv_regweibull(β, ρ, t, x)
    dα = dαdβ(β, x)
    ddf = ddlsurv_weibull(dot(β, x), ρ, t)
    #dfdβ = [dα[j] * df[1] for j in length(β)]   
    nb = length(β)
    np = nb + length(ρ)
    ddsdβ = zeros(np, np)
    ddsdβ[1:nb, np:np] .= ddf[1, 2] .* dα
    ddsdβ[np:np, 1:nb] .= ddf[2, 1] .* dα'
    ddsdβ[np, np] = ddf[2, 2]
    for r = 1:nb
        for c = r:nb
            ddsdβ[r, c] = ddsdβ[c, r] = ddf[1, 1] * dα[r] * dα[c]
        end
    end
    ddsdβ
end




##############################
# Exponential
##############################

mutable struct Exponential{T<:Real} <: AbstractSurvDist
    α::T   # shape (Weibull log-scale is 0.0)
end

function Exponential(α::T) where {T<:Int}
    Exponential(Float64(α))
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
Log probability distribution function: Exponential distribution

```julia
    β = [-2, 1.2]
    x = [2,.1]
    ρ = -0.5
    t = 3.0
    α = dot(β,x)
    d = Exponential()
    lpdf(d, t)
```
    
"""
function lpdf(d::Exponential, t)
    # location scale parameterization (Kalbfleisch and Prentice)
    #- d.α - t*exp(-d.α)
    lpdf_weibull(d.α, 0.0, t)
end


"""
Log survival function: Exponential distribution

```julia
    β = [-2, 1.2]
    x = [2,.1]
    ρ = -0.5
    t = 3.0
    α = dot(β,x)
    d = Exponential()
    lsurv(d, t)
```

"""
function lsurv(d::Exponential, t)
    # location scale parameterization (Kalbfleisch and Prentice)
    lsurv_weibull(d.α, 0.0, t)
end

"""
Log-likelihood calculation for Exponential regression: PDF

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Exponential()
lpdf_gradient(d, θ, t, x)
```
"""
function lpdf(d::Exponential, θ, t, x)
    lpdf_weibull(dot(θ,x), 0, t)
end

"""
Log-likelihood calculation for Exponential regression: Survival

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Exponential()
lsurv(d, θ, t, x)
```
"""
function lsurv(d::Exponential, θ, t, x)
    lsurv_weibull(dot(θ, x), 0, t)
end

"""
Gradient calculation for Exponential regression: PDF

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Exponential()
lpdf_gradient(d, θ, t, x)
```
"""
function lpdf_gradient(d::Exponential, θ, t, x)
    dlpdf_regweibull(θ, 0, t, x)[1:length(θ)]
end

"""
Gradient calculation for Exponential regression: Survival

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Exponential()
lsurv_gradient(d, θ, t, x)
```
"""
function lsurv_gradient(d::Exponential, θ, t, x)
    dlsurv_regweibull(θ, 0, t, x)[1:length(θ)]
end


"""
Hessian calculation for Exponential regression: PDF

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Exponential()
lpdf_hessian(d, θ, t, x)
```
"""
function lpdf_hessian(d::Exponential, θ, t, x)
    ddlpdf_regweibull(θ, 0, t, x)[1:length(θ), 1:length(θ)]
end

"""
Hessian calculation for Exponential regression: Survival
```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Exponential()
lsurv_hessian(d, θ, t, x)
```

"""
function lsurv_hessian(d::Exponential, θ, t, x)
    ddlsurv_regweibull(θ, 0, t, x)[1:length(θ), 1:length(θ)]
end

"""
Hessian calculation for Weibull distribution: PDF

```
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Exponential(α)
lpdf_hessian(d, t)
```
"""
function lpdf_hessian(d::Exponential, t)
    ddlpdf_weibull(d.α, 0.0, t)[1:1, 1:1]
end

"""
Hessian calculation for Exponential distribution: Survival

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Exponential(α)
lsurv_hessian(d, t)
```
"""
function lsurv_hessian(d::Exponential, t)
    ddlsurv_weibull(d.α, 0.0, t)[1:1, 1:1]
end


logscale(d::Exponential) = 0.0
scale(d::Exponential) = exp(logscale(d))
shape(d::Exponential) = d.α
params(d::Exponential) = (d.α,)



##################
# Lognormal
##################
struct Lognormal{T<:Real} <: AbstractSurvDist
    α::T   # shape: linear effects on this parameter
    ρ::T   # log-scale
end

function Lognormal(α::T, ρ::T) where {T<:Int}
    Lognormal(Float64(α), Float64(ρ))
end

function Lognormal(α::T, ρ::R) where {T<:Int,R<:Float64}
    Lognormal(Float64(α), ρ)
end

function Lognormal(α::R, ρ::T) where {R<:Float64,T<:Int}
    Lognormal(α, Float64(ρ))
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

Location scale representation (Klein Moeschberger ch 12)

"""
function lpdf(d::Lognormal, t)
    # location, log(scale) representation (Klein Moeschberger ch 12)
    lpdf_lognormal(d.α, d.ρ, t)
end

"""
log probability distribution function: Weibull distribution

Location scale representation (Klein Moeschberger ch 12)
"""
function lsurv(d::Lognormal, t)
    # location, log(scale) representation (Klein Moeschberger ch 12, modified from Wikipedia page 
    lsurv_lognormal(d.α, d.ρ, t)
end

"""
Log likelihood calculation for Lognormal regression: PDF

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Lognormal()
lpdf(d, vcat(θ,ρ), t, x)
```
"""
function lpdf(d::Lognormal, θ, t, x)
    lpdf_lognormal(dot(θ[1:end-1],x), θ[end], t)
end

"""
Log likelihood calculation for Log-normal regression: Survival

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Lognormal()
lsurv(d, vcat(θ,ρ), t, x)
```
"""
function lsurv(d::Lognormal, θ, t, x)
    lsurv_lognormal(dot(θ[1:end-1],x), θ[end], t)
end

"""
Gradient calculation for Lognormal regression: PDF

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Lognormal()
lpdf_gradient(d, vcat(θ,ρ), t, x)
```
"""
function lpdf_gradient(d::Lognormal, θ, t, x)
    dlpdf_reglognormal(θ[1:end-1], θ[end], t, x)
end

"""
Gradient calculation for Log-normal regression: Survival

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Lognormal()
lsurv_gradient(d, vcat(θ,ρ), t, x)
```
"""
function lsurv_gradient(d::Lognormal, θ, t, x)
    dlsurv_reglognormal(θ[1:end-1], θ[end], t, x)
end


"""
Hessian calculation for Log-normal regression: PDF

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Lognormal()
lpdf_hessian(d, vcat(θ,ρ), t, x)
```
"""
function lpdf_hessian(d::Lognormal, θ, t, x)
    ddlpdf_reglognormal(θ[1:end-1], θ[end], t, x)
end

"""
Hessian calculation for Log-normal regression: Survival
```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Lognormal()
lsurv_hessian(d, vcat(θ,ρ), t, x)
```

"""
function lsurv_hessian(d::Lognormal, θ, t, x)
    ddlsurv_reglognormal(θ[1:end-1], θ[end], t, x)
end

"""
Hessian calculation for Log-normal distribution: PDF

```
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Lognormal(α, ρ)
lpdf_hessian(d, t)
```
"""
function lpdf_hessian(d::Lognormal, t)
    ddlpdf_lognormal(d.α, d.ρ, t)
end

"""
Hessian calculation for Log-normal distribution: Survival

```julia
β = [-2, 1.2]
x = [2,.1]
ρ = -0.5
t = 3.0
α = dot(β,x)
d = Lognormal(α, ρ)
lsurv_hessian(d, t)
```
"""
function lsurv_hessian(d::Lognormal, t)
    ddlsurv_lognormal(d.α, d.ρ, t)
end




logscale(d::Lognormal) = d.ρ
scale(d::Lognormal) = exp(logscale(d))
shape(d::Lognormal) = d.α
params(d::Lognormal) = (d.α, d.ρ)

################################################
# underlying distribution functions, Log-normal distribution
################################################
function lpdf_lognormal(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    ret = -log(sqrt(2pi)) - log(t) - ρ
    ret += -z * z / 2.0
    ret
end

function lsurv_lognormal(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    ret = log(1 - cdfnorm(z))
    ret
end


################################################
# Underlying gradient, Lognormal distribution
################################################


function dlpdf_lognormal(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    [z*exp(-ρ),z^2 - 1]
end


function dlsurv_lognormal(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
        [inv(sqrt(2pi)) * exp(-0.5*z^2 - ρ) / (1 - cdfnorm(z)),
    z * inv(sqrt(2pi)) * exp(-0.5*z^2)  / (1 - cdfnorm(z))]
end

################################################
# Underlying Hessians, Lognormal distribution
################################################


function ddlpdf_lognormal(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    hess = zeros(2,2)
    hess[1] = -exp(-2ρ)
    hess[2] = hess[3] = -2 * z * exp(-ρ)
    hess[4] = -2*z^2
    hess
end

function ddlsurv_lognormal(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    cp = (1 - cdfnorm(z))
    q = cp*exp(ρ)
    w = exp(-0.5 * z^2)
    hess = zeros(2,2)
    hess[1] = inv(sqrt(2pi)) * w * z * exp(-ρ) / q - inv(2pi) *  exp(-z^2 - ρ) /  (q*cp)
    hess[2] = hess[3] = (-sqrt(2/pi)* w * ((-0.5*(z^2)))) / q - (inv(sqrt(2pi)) * w / q^2) * (q + inv(sqrt(2pi)) * w * exp(ρ) * z)
    hess[4] = (   z * w * ( inv(sqrt(2pi))*(1+z^2) - inv(sqrt(2pi)) * ( 2 + inv(sqrt(2pi)*q) * z * w * exp(ρ))))/cp
    hess
end


################################################
# Underlying gradient, Lognormal regression
################################################


function dlsurv_reglognormal(β, ρ, t, x)
    dα = dαdβ(β, x)
    df = dlsurv_lognormal(dot(β, x), ρ, t)
    dfdβ = [dα[j] * df[1] for j = 1:length(β)]
    dsdβ = vcat(dfdβ, df[2])
    dsdβ
end

function dlpdf_reglognormal(β, ρ, t, x)
    dα = dαdβ(β, x)
    df = dlpdf_lognormal(dot(β, x), ρ, t)
    dfdβ = [dα[j] * df[1] for j = 1:length(β)]
    dfdβ = vcat(dfdβ, df[2])
    dfdβ
end


################################################
# Underlying Hessians, Lognormal regression
################################################
# TODO: eliminate boilerplate

function ddlpdf_reglognormal(β, ρ, t, x)
    dα = dαdβ(β, x)
    ddf = ddlpdf_lognormal(dot(β, x), ρ, t)
    #dfdβ = [dα[j] * df[1] for j in length(β)]   
    nb = length(β)
    np = nb + length(ρ)
    ddsdβ = zeros(np, np)
    ddsdβ[1:nb, np:np] .= ddf[1, 2] .* dα
    ddsdβ[np:np, 1:nb] .= ddf[2, 1] .* dα'
    ddsdβ[np, np] = ddf[2, 2]
    for r = 1:nb
        for c = r:nb
            ddsdβ[r, c] = ddsdβ[c, r] = ddf[1, 1] * dα[r] * dα[c]
        end
    end
    ddsdβ
end

function ddlsurv_reglognormal(β, ρ, t, x)
    dα = dαdβ(β, x)
    ddf = ddlsurv_lognormal(dot(β, x), ρ, t)
    #dfdβ = [dα[j] * df[1] for j in length(β)]   
    nb = length(β)
    np = nb + length(ρ)
    ddsdβ = zeros(np, np)
    ddsdβ[1:nb, np:np] .= ddf[1, 2] .* dα
    ddsdβ[np:np, 1:nb] .= ddf[2, 1] .* dα'
    ddsdβ[np, np] = ddf[2, 2]
    for r = 1:nb
        for c = r:nb
            ddsdβ[r, c] = ddsdβ[c, r] = ddf[1, 1] * dα[r] * dα[c]
        end
    end
    ddsdβ
end


##############################
# Generalized Gamma distribution 
##############################
struct GGamma{R<:Real} <: AbstractSurvDist
    α::R   # shape: linear effects on this parameter
    ρ::R   # scale
    κ::R
end

function GGamma(α::T, ρ::T, κ::T) where {T<:Int}
    GGamma(Float64(α), Float64(ρ), Float64(κ))
end

function GGamma(α::T, ρ::T, κ::R) where {R<:Float64,T<:Int}
    GGamma(Float64(α), Float64(ρ), κ)
end

function GGamma(α::T, ρ::R, κ::T) where {T<:Int,R<:Float64}
    GGamma(Float64(α), ρ, Float64(κ))
end

function GGamma(α::R, ρ::T, κ::T) where {R<:Float64,T<:Int}
    GGamma(α, Float64(ρ), Float64(κ))
end

function GGamma(α::R, ρ::R, κ::T) where {R<:Float64,T<:Int}
    GGamma(α, ρ, Float64(κ))
end

function GGamma(α::R, ρ::T, κ::R) where {R<:Float64,T<:Int}
    GGamma(α, Float64(ρ), κ)
end

function GGamma(α::T, ρ::R, κ::R) where {R<:Float64,T<:Int}
    GGamma(Float64(α), ρ, κ)
end


function GGamma()
    GGamma(ones(Float64, 3)...)
end

# Methods for GGamma
"""
log probability distribution function, generalized gamma distribution

Location scale representation (Klein Moeschberger ch 12)

"""
function lpdf(d::GGamma, t)
    # location, log(scale) representation (Klein Moeschberger ch 12)
    lpdf_gengamma(d.α, d.ρ, d.κ, t)
end

"""
log probability distribution function, generalized gamma distribution

Location scale representation (Klein Moeschberger ch 12)
"""
function lsurv(d::GGamma, t)
    # location, log(scale) representation (Klein Moeschberger ch 12, modified from Wikipedia page 
    lsurv_gengamma(d.α, d.ρ, d.κ, t)
end

"""
log probability distribution for generalized gamma regression

"""
function lpdf(d::GGamma, θ, t, x)
    lpdf_gengamma(dot(θ[1:end-2],x), θ[end-1], θ[end], t)
end

"""
log survival distribution for generalized gamma regression
"""
function lsurv(d::GGamma, θ, t, x)
    lsurv_gengamma(dot(θ[1:end-2],x), θ[end-1], θ[end], t)
end

"""
log probability distribution gradient for generalized gamma regression
    analytic gradient
"""
function lpdf_gradient(d::GGamma, θ, t, x)
    dlpdf_reggengamma(θ[1:end-2], θ[end-1], θ[end], t, x)
end

"""
log survival distribution gradient for generalized gamma regression
    uses finite differences
"""
function lsurv_gradient(d::GGamma, θ, t, x)
    dlsurv_reggengamma(θ[1:end-2], θ[end-1], θ[end], t, x)
end


"""
Hessian calculation for generalized gamma regression: PDF

    placeholder function: returns nothing
"""
function lpdf_hessian(d::GGamma, θ, t, x)
    ddlpdf_reggengamma(θ[1:end-2], θ[end-1], θ[end], t, x)
end

"""
Hessian calculation for generalized gamma regression: Survival

    placeholder function: returns nothing
"""
function lsurv_hessian(d::GGamma, θ, t, x)
    ddlsurv_reggengamma(θ[1:end-2], θ[end-1], θ[end], t, x)
end

"""
Hessian calculation for generalized gamma distribution: PDF

placeholder function: returns nothing
"""
function lpdf_hessian(d::GGamma, t)
    ddlpdf_gengamma(d.α, d.ρ, d.κ, t)
end

"""
Hessian calculation for generalized gamma distribution: Survival

    placeholder function: returns nothing
"""
function lsurv_hessian(d::GGamma, t)
    ddlsurv_gengamma(d.α, d.ρ, d.κ, t)
end


logscale(d::GGamma) = d.ρ
scale(d::GGamma) = exp(logscale(d))
shape(d::GGamma) = d.α
params(d::GGamma) = (d.α, d.ρ, d.κ)

################################################
# underlying distribution functions, generalized gamma distribution
################################################
function lpdf_gengamma(α, ρ, κ, t)
    z = (log(t) - α) * exp(-ρ)
    z * exp(κ) - exp(z) - ρ - log(t) - loggamma(exp(κ))
end

function lsurv_gengamma(α, ρ, κ, t)
    z = (log(t) - α) * exp(-ρ)
    pkz, _ = gamma_inc(exp(κ), exp(z))
    log1p(-pkz)
end

################################################
# Underlying gradient, generalized gamma distribution
################################################

"""
α=0.1
ρ =-1.2
κ=1.9
t = 2.0

exp((log(t) - α)*exp(-ρ) - ρ) - exp(κ - ρ)
(α - log(t))*exp(κ - ρ) + (log(t) - α)*exp((log(t) - α)*exp(-ρ) - ρ) - 1
(log(t) - α)*exp(κ - ρ) - exp(κ)*SpecialFunctions.digamma(exp(κ))
"""
function dlpdf_gengamma(α, ρ, κ, t)
    z = (log(t) - α) * exp(-ρ)

    [exp(z - ρ) - exp(κ - ρ),
    z * (exp(z) - exp(κ)) - 1.0,
    (z - digamma(exp(κ)))*exp(κ)
    ]
end


"""
α=0.1
ρ =-1.2
κ=1.9
t = 2.0
dlsurv_gengamma(α, ρ, κ, t; fd = 1e-14)
"""
function dlsurv_gengamma(α, ρ, κ, t; fd = 1e-14)
    fdr = abs.([α,ρ,κ].*fd)
    dirs = [-1.0, 1.0]
    [
    diff(lsurv_gengamma.(α.+fdr[1]/2.0.*dirs, ρ, κ, t))[1]/fdr[1],
    diff(lsurv_gengamma.(α, ρ.+fdr[2]/2.0.*dirs, κ, t))[1]/fdr[2],
    diff(lsurv_gengamma.(α, ρ, κ.+fdr[3]/2.0.*dirs, t))[1]/fdr[3]
    ]
end

################################################
# Underlying Hessians, gengamma distribution
################################################


function ddlpdf_gengamma(α, ρ, κ, t)
    nothing
end

function ddlsurv_gengamma(α, ρ, κ, t)
    nothing
end


################################################
# Underlying gradient, gengamma regression
################################################


function dlsurv_reggengamma(β, ρ, κ, t, x)
    dα = dαdβ(β, x)
    df = dlsurv_gengamma(dot(β, x), ρ, κ, t)
    dfdβ = [dα[j] * df[1] for j = 1:length(β)]
    dsdβ = vcat(dfdβ, df[2:3])
    dsdβ
end

function dlpdf_reggengamma(β, ρ, κ, t, x)
    dα = dαdβ(β, x)
    df = dlpdf_gengamma(dot(β, x), ρ, κ, t)
    dfdβ = [dα[j] * df[1] for j = 1:length(β)]
    dfdβ = vcat(dfdβ, df[2:3])
    dfdβ
end


################################################
# Underlying Hessians, gengamma regression
################################################
# TODO: eliminate boilerplate

function ddlpdf_reggengamma(β, ρ, κ, t, x)
    nothing
end

function ddlsurv_reggengamma(β, ρ, κ, t, x)
    nothing
end

##############################
# Gamma distribution 
##############################
struct Gamma{T<:Real} <: AbstractSurvDist
    α::T   # shape: linear effects on this parameter
    κ::T
end

function Gamma(α::T, κ::T) where {T<:Int}
    Gamma(Float64(α), Float64(κ))
end

function Gamma(α::T, κ::R) where {R<:Float64,T<:Int}
    Gamma(Float64(α), κ)
end

function Gamma(α::R, κ::T) where {R<:Float64,T<:Int}
    Gamma(α,  Float64(κ))
end

function Gamma()
    Gamma(ones(Float64, 2)...)
end

# Methods for Gamma
"""
log probability distribution function, Gamma distribution

Location scale representation (Klein Moeschberger ch 12)

"""
function lpdf(d::Gamma, t)
    # location, log(scale) representation (Klein Moeschberger ch 12)
    lpdf_gamma(d.α, d.κ, t)
end

"""
log probability distribution function, Gamma distribution

Location scale representation (Klein Moeschberger ch 12)
"""
function lsurv(d::Gamma, t)
    lsurv_gamma(d.α, d.κ, t)
end

"""
log probability distribution for Gamma regression

"""
function lpdf(d::Gamma, θ, t, x)
    lpdf_gamma(dot(θ[1:end-1],x), θ[end], t)
end

"""
log survival distribution for Gamma regression
"""
function lsurv(d::Gamma, θ, t, x)
    lsurv_gamma(dot(θ[1:end-1],x), θ[end], t)
end

"""
log probability distribution gradient for Gamma regression
    analytic gradient
"""
function lpdf_gradient(d::Gamma, θ, t, x)
    dlpdf_reggamma(θ[1:end-1], θ[end], t, x)
end

"""
log survival distribution gradient for Gamma regression
    uses finite differences
"""
function lsurv_gradient(d::Gamma, θ, t, x)
    dlsurv_reggamma(θ[1:end-1], θ[end], t, x)
end


"""
Hessian calculation for Gamma regression: PDF

    placeholder function: returns nothing
"""
function lpdf_hessian(d::Gamma, θ, t, x)
    ddlpdf_reggamma(θ[1:end-1], θ[end], t, x)
end

"""
Hessian calculation for Gamma regression: Survival

    placeholder function: returns nothing
"""
function lsurv_hessian(d::Gamma, θ, t, x)
    ddlsurv_reggamma(θ[1:end-1], θ[end], t, x)
end

"""
Hessian calculation for Gamma distribution: PDF

placeholder function: returns nothing
"""
function lpdf_hessian(d::Gamma, t)
    ddlpdf_gamma(d.α, d.κ, t)
end

"""
Hessian calculation for Gamma distribution: Survival

    placeholder function: returns nothing
"""
function lsurv_hessian(d::Gamma, t)
    ddlsurv_gamma(d.α, d.κ, t)
end


logscale(d::Gamma) = 0.0
scale(d::Gamma) = exp(logscale(d))
shape(d::Gamma) = d.α
params(d::Gamma) = (d.α, d.κ)

################################################
# underlying distribution functions, Gamma distribution
################################################
function lpdf_gamma(α, κ, t)
    zs = (log(t) - α)
    zs * exp(κ) - exp(zs) - log(t) - loggamma(exp(κ))
end

function lsurv_gamma(α, κ, t)
    zs = (log(t) - α)
    pkz, _ = gamma_inc(exp(κ), exp(zs))
    log1p(-pkz)
end

################################################
# Underlying gradient, Gamma distribution
################################################


function dlpdf_gamma(α, κ, t)
    z = (log(t) - α)

    [exp(z) - exp(κ),
    (z - digamma(exp(κ)))*exp(κ)
    ]
end



function dlsurv_gamma(α, κ, t; fd = 1e-14)
    fdr = abs.([α, κ].*fd)
    dirs = [-1.0, 1.0]
    [
    diff(lsurv_gamma.(α.+fdr[1]/2.0.*dirs, κ, t))[1]/fdr[1],
    diff(lsurv_gamma.(α, κ.+fdr[2]/2.0.*dirs, t))[1]/fdr[2]
    ]
end

################################################
# Underlying Hessians, gamma distribution
################################################


function ddlpdf_gamma(α, κ, t)
    nothing
end

function ddlsurv_gamma(α, κ, t)
    nothing
end


################################################
# Underlying gradient, gamma regression
################################################


function dlsurv_reggamma(β, κ, t, x)
    dα = dαdβ(β, x)
    df = dlsurv_gamma(dot(β, x), κ, t)
    dfdβ = [dα[j] * df[1] for j = 1:length(β)]
    dsdβ = vcat(dfdβ, df[2])
    dsdβ
end

function dlpdf_reggamma(β, κ, t, x)
    dα = dαdβ(β, x)
    df = dlpdf_gamma(dot(β, x), κ, t)
    dfdβ = [dα[j] * df[1] for j = 1:length(β)]
    dfdβ = vcat(dfdβ, df[2])
    dfdβ
end


################################################
# Underlying Hessians, gamma regression
################################################
# TODO: eliminate boilerplate

function ddlpdf_reggamma(β, κ, t, x)
    nothing
end

function ddlsurv_reggamma(β, κ, t, x)
    nothing
end