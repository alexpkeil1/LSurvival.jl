


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

        Note that there is no checking that parameters α,ρ are positively bound, and p ∈ (0,1), and errors will be given if this is not the case

Signature:

```julia        
qweibull(p::Real,α::Real,ρ::Real)
```
# quantile(Weibull(.75, 1.1), .3) # cross reference the approach in the Distributions package
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
"""
randweibull(rng, α, ρ) = qweibull(rand(rng), α, ρ)
randweibull(α, ρ) = randweibull(MersenneTwister(), α, ρ)

######################################################################
# Distributions used in parametric survival models
######################################################################


raw"""
Mean model parameter gradient function for linear model
    $f(\theta, X) = \theta X$
"""
function dαdθ(θ, X)
    [x for x in X]
end



##############################
# Weibull distribution 
##############################
struct Weibull{T<:Real} <: AbstractSurvDist
    α::T   # scale: linear effects on this parameter
    ρ::T   # shape
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
shape(d::Weibull) = d.ρ
scale(d::Weibull) = d.α
params(d::Weibull) = (d.α, d.ρ)

################################################
# underlying distribution functions, Weibull distribution
################################################
#=
# pg 34 of Kalbfleisch and Prentice
$$f(t;\rho,\beta) = \exp(\gamma)^{-1}\exp(\ln(t)-\rho)/\exp(\gamma))\exp(-\exp((\ln(t)-\rho)/\exp(\gamma)))$$
$$\ln\mathfrak{L}(t;\rho,\beta) = -2\gamma + \ln(t)-\rho  -\exp((\ln(t)-\rho)\exp(-\gamma))$$

=#

function lpdf_weibull(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    ret = -ρ + z - exp(z) - log(t)
    ret
end

function lsurv_weibull(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    ret = -exp(z)
    ret
end



################################################
# underlying gradients, Weibull distribution
################################################
function dlpdf_weibull(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    [(exp(z) - 1.0) / exp(ρ), z * exp(z) - z - 1]
end

function dlsurv_weibull(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    [exp(z) / exp(ρ), z * exp(z)]
end


################################################
# underlying Hessians, Weibull distribution
################################################

function ddlpdf_weibull(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    vcat(
        hcat((-exp(z)) / (exp(ρ)^2), exp(-ρ) - (z + 1.0) * exp(z - ρ)),
        hcat(exp(-ρ) - (z + 1.0) * exp(z - ρ), z - z * exp(z) - z * exp(z)),
    )
end

function ddlsurv_weibull(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    vcat(
        hcat(-exp(z - 2ρ), -(z + 1) * exp(z - ρ)),
        hcat(-(z + 1) * exp(z - ρ), -z * exp(z) - z^2 * exp(z)),
    )
end


################################################
# Underlying gradients, Weibull regression
################################################

function dlsurv_regweibull(θ, ρ, t, x)
    dα = dαdθ(θ, x)
    df = dlsurv_weibull(dot(θ, x), ρ, t)
    dfdθ = [dα[j] * df[1] for j = 1:length(θ)]
    dsdθ = vcat(dfdθ, df[2])
    dsdθ
end

function dlpdf_regweibull(θ, ρ, t, x)
    dα = dαdθ(θ, x)
    df = dlpdf_weibull(dot(θ, x), ρ, t)
    dfdθ = [dα[j] * df[1] for j = 1:length(θ)]
    dfdθ = vcat(dfdθ, df[2])
    dfdθ
end


################################################
# Underlying Hessians, Weibull regression
################################################

function ddlpdf_regweibull(θ, ρ, t, x)
    dα = dαdθ(θ, x)
    ddf = ddlpdf_weibull(dot(θ, x), ρ, t)
    #dfdθ = [dα[j] * df[1] for j in length(θ)]   
    nb = length(θ)
    np = nb + length(ρ)
    ddsdθ = zeros(np, np)
    ddsdθ[1:nb, np:np] .= ddf[1, 2] .* dα
    ddsdθ[np:np, 1:nb] .= ddf[2, 1] .* dα'
    ddsdθ[np, np] = ddf[2, 2]
    for r = 1:nb
        for c = r:nb
            ddsdθ[r, c] = ddsdθ[c, r] = ddf[1, 1] * dα[r] * dα[c]
        end
    end
    ddsdθ
end

function ddlsurv_regweibull(θ, ρ, t, x)
    dα = dαdθ(θ, x)
    ddf = ddlsurv_weibull(dot(θ, x), ρ, t)
    #dfdθ = [dα[j] * df[1] for j in length(θ)]   
    nb = length(θ)
    np = nb + length(ρ)
    ddsdθ = zeros(np, np)
    ddsdθ[1:nb, np:np] .= ddf[1, 2] .* dα
    ddsdθ[np:np, 1:nb] .= ddf[2, 1] .* dα'
    ddsdθ[np, np] = ddf[2, 2]
    for r = 1:nb
        for c = r:nb
            ddsdθ[r, c] = ddsdθ[c, r] = ddf[1, 1] * dα[r] * dα[c]
        end
    end
    ddsdθ
end




##############################
# Exponential
##############################

mutable struct Exponential{T<:Real} <: AbstractSurvDist
    α::T   # scale (Weibull shape is 1.0)
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
    lpdf_weibull(d.α, 1, t)
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
    lsurv_weibull(d.α, 1, t)
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

shape(d::Exponential) = 1.0
scale(d::Exponential) = d.ρ
params(d::Exponential) = (d.ρ,)



##################
# Lognormal
##################
struct Lognormal{T<:Real} <: AbstractSurvDist
    α::T   # scale: linear effects on this parameter
    ρ::T   # shape
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



shape(d::Lognormal) = d.ρ
scale(d::Lognormal) = d.α
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


function dlsurv_reglognormal(θ, ρ, t, x)
    dα = dαdθ(θ, x)
    df = dlsurv_lognormal(dot(θ, x), ρ, t)
    dfdθ = [dα[j] * df[1] for j = 1:length(θ)]
    dsdθ = vcat(dfdθ, df[2])
    dsdθ
end

function dlpdf_reglognormal(θ, ρ, t, x)
    dα = dαdθ(θ, x)
    df = dlpdf_lognormal(dot(θ, x), ρ, t)
    dfdθ = [dα[j] * df[1] for j = 1:length(θ)]
    dfdθ = vcat(dfdθ, df[2])
    dfdθ
end


################################################
# Underlying Hessians, Lognormal regression
################################################
# TODO: eliminate boilerplate

function ddlpdf_reglognormal(θ, ρ, t, x)
    dα = dαdθ(θ, x)
    ddf = ddlpdf_lognormal(dot(θ, x), ρ, t)
    #dfdθ = [dα[j] * df[1] for j in length(θ)]   
    nb = length(θ)
    np = nb + length(ρ)
    ddsdθ = zeros(np, np)
    ddsdθ[1:nb, np:np] .= ddf[1, 2] .* dα
    ddsdθ[np:np, 1:nb] .= ddf[2, 1] .* dα'
    ddsdθ[np, np] = ddf[2, 2]
    for r = 1:nb
        for c = r:nb
            ddsdθ[r, c] = ddsdθ[c, r] = ddf[1, 1] * dα[r] * dα[c]
        end
    end
    ddsdθ
end

function ddlsurv_reglognormal(θ, ρ, t, x)
    dα = dαdθ(θ, x)
    ddf = ddlsurv_lognormal(dot(θ, x), ρ, t)
    #dfdθ = [dα[j] * df[1] for j in length(θ)]   
    nb = length(θ)
    np = nb + length(ρ)
    ddsdθ = zeros(np, np)
    ddsdθ[1:nb, np:np] .= ddf[1, 2] .* dα
    ddsdθ[np:np, 1:nb] .= ddf[2, 1] .* dα'
    ddsdθ[np, np] = ddf[2, 2]
    for r = 1:nb
        for c = r:nb
            ddsdθ[r, c] = ddsdθ[c, r] = ddf[1, 1] * dα[r] * dα[c]
        end
    end
    ddsdθ
end