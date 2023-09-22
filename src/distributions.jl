


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
cdfnorm(z) = 0.5 * (1 + erf(z / sqrt(2)))


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


raw"""
Mean model parameter gradient function for linear model
    $f(\theta, X) = \theta X$
"""
function dρdθ(θ, X)
    [x for x in X]
end



##############################
# Weibull distribution 
##############################
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
Log probability distribution function: Weibull distribution

# location scale representation (Klein Moeschberger ch 12)
# Lik: λγ(λt)^{γ-1}exp(-(λt)^γ) # traditional Weibull
# Lik: exp(-α)(1/σ) (exp(-α)exp(log(t)))^{(1/σ)-1} exp(-(exp(-α)exp(log(t)))^(1/σ))           , γ=σ^-1, λ=exp(-α), t = exp(log(t))
# Lik: (1/σ)exp(-α)  exp(log(t)-α)^{(1/σ)-1} exp(-(exp(log(t)-α)/σ))           
# Lik: 1/σ *exp(-α) exp((log(t)-α)/σ-(log(t)-α)) exp(-(exp(log(t)-α)/σ))           
# lLik: log(1/σ) - α +  (log(t)-α)/σ -log(t)+α       - exp(log(t)-α)/σ)           
# lLik: log(1/σ)     +  z            -log(t)         - exp(z)                 z = (log(t)-α)/σ
# lLik: -log(σ) + z - exp(z) - log(t),                                        σ=d.γ=γ^-1, α=d.ρ=-log(λ)

```julia
ρ=0.1   # location
γ=-1.2  # log(scale)
time=2
lpdf(Weibull(ρ, γ), time)
```
"""
function lpdf(d::Weibull, t)
    lpdf_weibull(d.ρ, d.γ, t)
end



"""
Log survival distribution function: Weibull distribution
    
# location, log(scale) representation (Klein Moeschberger ch 12)
# Surv: exp(-(λt)^γ) # traditional Weibull
# Surv: exp(-(exp(-α)exp(log(t)))^(1/σ)) , γ=exp(σ)^-1, λ=exp(-α), t = exp(log(t))
# Surv: exp(-exp((log(t)-α)/exp(σ))) 
# lSurv: -exp((log(t)-α)/exp(σ))
# lSurv: -exp(z)

```julia
ρ=0.1   # location
γ=-1.2  # log(scale)
time=2
lsurv(Weibull(ρ, γ), time)
```

"""
function lsurv(d::Weibull, t)
    # location scale representation (Klein Moeschberger ch 12, modified from Wikipedia page on Gumbel Distribution)
    lsurv_weibull(d.ρ, d.γ, t)
end


"""
Log-likelihood calculation for weibull regression: PDF

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Weibull()
lpdf(d, vcat(θ,γ), t, x)
```
"""
function lpdf(d::Weibull, θ, t, x)
    lpdf_weibull(dot(θ[1:end-1], x), θ[end], t)
end

"""
Log-likelihood calculation for weibull regression: Survival

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Weibull()
lsurv(d, vcat(θ,γ), t, x)
```
"""
function lsurv(d::Weibull, θ, t, x)
    lsurv_weibull(dot(θ[1:end-1], x), θ[end], t)
end

"""
Gradient calculation for weibull regression: PDF

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
#ρ = dot(θ,x)
d = Weibull()
lpdf_gradient(d, vcat(θ,γ), t, x)
```
"""
function lpdf_gradient(d::Weibull, θ, t, x)
    dlpdf_regweibull(θ[1:end-1], θ[end], t, x)
end

"""
Gradient calculation for weibull regression: Survival

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Weibull()
lsurv_gradient(d, vcat(θ,γ), t, x)
```
"""
function lsurv_gradient(d::Weibull, θ, t, x)
    dlsurv_regweibull(θ[1:end-1], θ[end], t, x)
end


"""
Hessian calculation for weibull regression: PDF

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Weibull()
lpdf_hessian(d, vcat(θ,γ), t, x)
```
"""
function lpdf_hessian(d::Weibull, θ, t, x)
    ddlpdf_regweibull(θ[1:end-1], θ[end], t, x)
end

"""
Hessian calculation for weibull regression: Survival
```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Weibull()
lsurv_hessian(d, vcat(θ,γ), t, x)
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
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Weibull(ρ, γ)
lpdf_hessian(d, t)
```
"""
function lpdf_hessian(d::Weibull, t)
    ddlpdf_weibull(d.ρ, d.γ, t)
end

"""
Hessian calculation for Weibull distribution: Survival

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Weibull(ρ, γ)
lsurv_hessian(d, t)
```
"""
function lsurv_hessian(d::Weibull, t)
    ddlsurv_weibull(d.ρ, d.γ, t)
end


shape(d::Weibull) = d.γ
scale(d::Weibull) = d.ρ
params(d::Weibull) = (d.ρ, d.γ)

################################################
# underlying distribution functions, Weibull distribution
################################################
#=
# pg 34 of Kalbfleisch and Prentice
$$f(t;\rho,\beta) = \exp(\gamma)^{-1}\exp(\ln(t)-\rho)/\exp(\gamma))\exp(-\exp((\ln(t)-\rho)/\exp(\gamma)))$$
$$\ln\mathfrak{L}(t;\rho,\beta) = -2\gamma + \ln(t)-\rho  -\exp((\ln(t)-\rho)\exp(-\gamma))$$

=#

function lpdf_weibull(ρ, γ, t)
    z = (log(t) - ρ) / exp(γ)
    ret = -γ + z - exp(z) - log(t)
    ret
end

function lsurv_weibull(ρ, γ, t)
    z = (log(t) - ρ) / exp(γ)
    ret = -exp(z)
    ret
end



################################################
# underlying gradients, Weibull distribution
################################################
function dlpdf_weibull(ρ, γ, t)
    z = (log(t) - ρ) / exp(γ)
    [(exp(z) - 1.0) / exp(γ), z * exp(z) - z - 1]
end

function dlsurv_weibull(ρ, γ, t)
    z = (log(t) - ρ) / exp(γ)
    [exp(z) / exp(γ), z * exp(z)]
end


################################################
# underlying Hessians, Weibull distribution
################################################

function ddlpdf_weibull(ρ, γ, t)
    z = (log(t) - ρ) / exp(γ)
    vcat(
        hcat((-exp(z)) / (exp(γ)^2), exp(-γ) - (z + 1.0) * exp(z - γ)),
        hcat(exp(-γ) - (z + 1.0) * exp(z - γ), z - z * exp(z) - z * exp(z)),
    )
end

function ddlsurv_weibull(ρ, γ, t)
    z = (log(t) - ρ) / exp(γ)
    vcat(
        hcat(-exp(z - 2γ), -(z + 1) * exp(z - γ)),
        hcat(-(z + 1) * exp(z - γ), -z * exp(z) - z^2 * exp(z)),
    )
end


################################################
# Underlying gradients, Weibull regression
################################################

function dlsurv_regweibull(θ, γ, t, x)
    dρ = dρdθ(θ, x)
    df = dlsurv_weibull(dot(θ, x), γ, t)
    dfdθ = [dρ[j] * df[1] for j = 1:length(θ)]
    dsdθ = vcat(dfdθ, df[2])
    dsdθ
end

function dlpdf_regweibull(θ, γ, t, x)
    dρ = dρdθ(θ, x)
    df = dlpdf_weibull(dot(θ, x), γ, t)
    dfdθ = [dρ[j] * df[1] for j = 1:length(θ)]
    dfdθ = vcat(dfdθ, df[2])
    dfdθ
end


################################################
# Underlying Hessians, Weibull regression
################################################

function ddlpdf_regweibull(θ, γ, t, x)
    dρ = dρdθ(θ, x)
    ddf = ddlpdf_weibull(dot(θ, x), γ, t)
    #dfdθ = [dρ[j] * df[1] for j in length(θ)]   
    nb = length(θ)
    np = nb + length(γ)
    ddsdθ = zeros(np, np)
    ddsdθ[1:nb, np:np] .= ddf[1, 2] .* dρ
    ddsdθ[np:np, 1:nb] .= ddf[2, 1] .* dρ'
    ddsdθ[np, np] = ddf[2, 2]
    for r = 1:nb
        for c = r:nb
            ddsdθ[r, c] = ddsdθ[c, r] = ddf[1, 1] * dρ[r] * dρ[c]
        end
    end
    ddsdθ
end

function ddlsurv_regweibull(θ, γ, t, x)
    dρ = dρdθ(θ, x)
    ddf = ddlsurv_weibull(dot(θ, x), γ, t)
    #dfdθ = [dρ[j] * df[1] for j in length(θ)]   
    nb = length(θ)
    np = nb + length(γ)
    ddsdθ = zeros(np, np)
    ddsdθ[1:nb, np:np] .= ddf[1, 2] .* dρ
    ddsdθ[np:np, 1:nb] .= ddf[2, 1] .* dρ'
    ddsdθ[np, np] = ddf[2, 2]
    for r = 1:nb
        for c = r:nb
            ddsdθ[r, c] = ddsdθ[c, r] = ddf[1, 1] * dρ[r] * dρ[c]
        end
    end
    ddsdθ
end




##############################
# Exponential
##############################

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
Log probability distribution function: Exponential distribution

    ```julia
    θ = [-2, 1.2]
    x = [2,.1]
    γ = -0.5
    t = 3.0
    ρ = dot(θ,x)
    d = Exponential()
    lpdf(d, t)
    ```
    
"""
function lpdf(d::Exponential, t)
    # location scale parameterization (Kalbfleisch and Prentice)
    #- d.ρ - t*exp(-d.ρ)
    lpdf_weibull(d.ρ, 1, t)
end


"""
Log survival function: Exponential distribution

    ```julia
    θ = [-2, 1.2]
    x = [2,.1]
    γ = -0.5
    t = 3.0
    ρ = dot(θ,x)
    d = Exponential()
    lsurv(d, t)
    ```

"""
function lsurv(d::Exponential, t)
    # location scale parameterization (Kalbfleisch and Prentice)
    lsurv_weibull(d.ρ, 1, t)
end

"""
Log-likelihood calculation for Exponential regression: PDF

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
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
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
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
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
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
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
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
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
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
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
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
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Exponential(ρ)
lpdf_hessian(d, t)
```
"""
function lpdf_hessian(d::Exponential, t)
    ddlpdf_weibull(d.ρ, 0.0, t)[1:1, 1:1]
end

"""
Hessian calculation for Exponential distribution: Survival

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Exponential(ρ)
lsurv_hessian(d, t)
```
"""
function lsurv_hessian(d::Exponential, t)
    ddlsurv_weibull(d.ρ, 0.0, t)[1:1, 1:1]
end

shape(d::Exponential) = 1.0
scale(d::Exponential) = d.γ
params(d::Exponential) = (d.γ,)



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
# Lik: (2π)^(-1/2)(1/exp(σ))/exp(log(t)) exp((-(1/exp(σ))^2 log(exp(-α)exp(log(t)))^2)/2)           , γ=(1/exp(σ)), λ=exp(-α), t = exp(log(t))
# Lik: (2π)^(-1/2)(1/(exp(σ)t)) exp((-(1/exp(σ))^2 log(exp(log(t)-α))^2)/2)
# Lik: (2π)^(-1/2)(1/(exp(σ)t)) exp((-1/2 * (log(t)-α)/exp(σ))^2))
# Lik: (2π)^(-1/2)(exp(-σ)1/t) exp((-1/2 * z^2))
# lLik: log(-sqrt(2π) exp(-σ) 1/t)) -z^2/2                                        
# lLik: log(-sqrt(2π))) - log(t) - σ - z^2/2                                        σ=d.γ=γ^-1, α=d.ρ=-log(λ)

"""
function lpdf(d::Lognormal, t)
    # location, log(scale) representation (Klein Moeschberger ch 12)
    lpdf_lognormal(d.ρ, d.γ, t)
end

"""
log probability distribution function: Weibull distribution

# location scale representation (Klein Moeschberger ch 12)
# Surv: 1- Φ(γlog(λt))      # traditional Lognormal
# Surv: 1- Φ((1/exp(σ))log(exp(-α)exp(log(t))))                        , γ=(1/exp(σ)), λ=exp(-α), t = exp(log(t))
# Surv: 1- Φ((log(t)-α)/exp(σ))
# iSurv: log(1-Φ(z))                                        σ=d.γ=γ^-1, α=d.ρ=-log(λ)
"""
function lsurv(d::Lognormal, t)
    # location, log(scale) representation (Klein Moeschberger ch 12, modified from Wikipedia page 
    lsurv_lognormal(d.ρ, d.γ, t)
end

"""
Log likelihood calculation for Lognormal regression: PDF

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Lognormal()
lpdf(d, vcat(θ,γ), t, x)
```
"""
function lpdf(d::Lognormal, θ, t, x)
    lpdf_lognormal(dot(θ[1:end-1],x), θ[end], t)
end

"""
Log likelihood calculation for Log-normal regression: Survival

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Lognormal()
lsurv(d, vcat(θ,γ), t, x)
```
"""
function lsurv(d::Lognormal, θ, t, x)
    lsurv_lognormal(dot(θ[1:end-1],x), θ[end], t)
end

"""
Gradient calculation for Lognormal regression: PDF

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Lognormal()
lpdf_gradient(d, vcat(θ,γ), t, x)
```
"""
function lpdf_gradient(d::Lognormal, θ, t, x)
    dlpdf_reglognormal(θ[1:end-1], θ[end], t, x)
end

"""
Gradient calculation for Log-normal regression: Survival

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Lognormal()
lsurv_gradient(d, vcat(θ,γ), t, x)
```
"""
function lsurv_gradient(d::Lognormal, θ, t, x)
    dlsurv_reglognormal(θ[1:end-1], θ[end], t, x)
end


"""
Hessian calculation for Log-normal regression: PDF

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Lognormal()
lpdf_hessian(d, vcat(θ,γ), t, x)
```
"""
function lpdf_hessian(d::Lognormal, θ, t, x)
    ddlpdf_reglognormal(θ[1:end-1], θ[end], t, x)
end

"""
Hessian calculation for Log-normal regression: Survival
```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Lognormal()
lsurv_hessian(d, vcat(θ,γ), t, x)
```

"""
function lsurv_hessian(d::Lognormal, θ, t, x)
    ddlsurv_reglognormal(θ[1:end-1], θ[end], t, x)
end

"""
Hessian calculation for Log-normal distribution: PDF

```
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Lognormal(ρ, γ)
lpdf_hessian(d, t)
```
"""
function lpdf_hessian(d::Lognormal, t)
    ddlpdf_lognormal(d.ρ, d.γ, t)
end

"""
Hessian calculation for Log-normal distribution: Survival

```julia
θ = [-2, 1.2]
x = [2,.1]
γ = -0.5
t = 3.0
ρ = dot(θ,x)
d = Lognormal(ρ, γ)
lsurv_hessian(d, t)
```
"""
function lsurv_hessian(d::Lognormal, t)
    ddlsurv_lognormal(d.ρ, d.γ, t)
end



shape(d::Lognormal) = d.γ
scale(d::Lognormal) = d.ρ
params(d::Lognormal) = (d.ρ, d.γ)


################################################
# underlying distribution functions, Log-normal distribution
################################################
function lpdf_lognormal(ρ, γ, t)
    z = (log(t) - ρ) / exp(γ)
    ret = -log(sqrt(2pi)) - log(t) - γ
    ret += -z * z / 2.0
    ret
end

function lsurv_lognormal(ρ, γ, t)
    z = (log(t) - ρ) / exp(γ)
    ret = log(1 - cdfnorm(z))
    ret
end


################################################
# Underlying gradient, Lognormal distribution
################################################


function dlpdf_lognormal(ρ, γ, t)
    z = (log(t) - ρ) / exp(γ)
    [z*exp(-γ),z^2 - 1]
end


function dlsurv_lognormal(ρ, γ, t)
    z = (log(t) - ρ) / exp(γ)
        [inv(sqrt(2pi)) * exp(-0.5*z^2 - γ) / (1 - cdfnorm(z)),
    z * inv(sqrt(2pi)) * exp(-0.5*z^2)  / (1 - cdfnorm(z))]
end

################################################
# Underlying Hessians, Lognormal distribution
################################################


function ddlpdf_lognormal(ρ, γ, t)
    z = (log(t) - ρ) / exp(γ)
    hess = zeros(2,2)
    hess[1] = -exp(-2γ)
    hess[2] = hess[3] = -2 * z * exp(-γ)
    hess[4] = -2*z^2
    hess
end

function ddlsurv_lognormal(ρ, γ, t)
    z = (log(t) - ρ) / exp(γ)
    cp = (1 - cdfnorm(z))
    q = cp*exp(γ)
    w = exp(-0.5 * z^2)
    hess = zeros(2,2)
    hess[1] = inv(sqrt(2pi)) * w * z * exp(-γ) / q - inv(2pi) *  exp(-z^2 - γ) /  (q*cp)
    hess[2] = hess[3] = (-sqrt(2/pi)* w * ((-0.5*(z^2)))) / q - (inv(sqrt(2pi)) * w / q^2) * (q + inv(sqrt(2pi)) * w * exp(γ) * z)
    hess[4] = (   z * w * ( inv(sqrt(2pi))*(1+z^2) - inv(sqrt(2pi)) * ( 2 + inv(sqrt(2pi)*q) * z * w * exp(γ))))/cp
    hess
end


################################################
# Underlying gradient, Lognormal regression
################################################


function dlsurv_reglognormal(θ, γ, t, x)
    dρ = dρdθ(θ, x)
    df = dlsurv_lognormal(dot(θ, x), γ, t)
    dfdθ = [dρ[j] * df[1] for j = 1:length(θ)]
    dsdθ = vcat(dfdθ, df[2])
    dsdθ
end

function dlpdf_reglognormal(θ, γ, t, x)
    dρ = dρdθ(θ, x)
    df = dlpdf_lognormal(dot(θ, x), γ, t)
    dfdθ = [dρ[j] * df[1] for j = 1:length(θ)]
    dfdθ = vcat(dfdθ, df[2])
    dfdθ
end


################################################
# Underlying Hessians, Lognormal regression
################################################
# TODO: eliminate boilerplate

function ddlpdf_reglognormal(θ, γ, t, x)
    dρ = dρdθ(θ, x)
    ddf = ddlpdf_lognormal(dot(θ, x), γ, t)
    #dfdθ = [dρ[j] * df[1] for j in length(θ)]   
    nb = length(θ)
    np = nb + length(γ)
    ddsdθ = zeros(np, np)
    ddsdθ[1:nb, np:np] .= ddf[1, 2] .* dρ
    ddsdθ[np:np, 1:nb] .= ddf[2, 1] .* dρ'
    ddsdθ[np, np] = ddf[2, 2]
    for r = 1:nb
        for c = r:nb
            ddsdθ[r, c] = ddsdθ[c, r] = ddf[1, 1] * dρ[r] * dρ[c]
        end
    end
    ddsdθ
end

function ddlsurv_reglognormal(θ, γ, t, x)
    dρ = dρdθ(θ, x)
    ddf = ddlsurv_lognormal(dot(θ, x), γ, t)
    #dfdθ = [dρ[j] * df[1] for j in length(θ)]   
    nb = length(θ)
    np = nb + length(γ)
    ddsdθ = zeros(np, np)
    ddsdθ[1:nb, np:np] .= ddf[1, 2] .* dρ
    ddsdθ[np:np, 1:nb] .= ddf[2, 1] .* dρ'
    ddsdθ[np, np] = ddf[2, 2]
    for r = 1:nb
        for c = r:nb
            ddsdθ[r, c] = ddsdθ[c, r] = ddf[1, 1] * dρ[r] * dρ[c]
        end
    end
    ddsdθ
end