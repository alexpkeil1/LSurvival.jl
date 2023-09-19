# Parametric survival models
######################################################################
# Survival model distributions
######################################################################

#abstract type AbstractSurvDist end

function copydist(d::T, args...) where {T<:AbstractSurvDist}
    typeof(d)(args...)
end

function Base.length(d::T) where {T<:AbstractSurvDist}
    length(fieldnames(T))
end

function name(::Type{T}) where {T}
    isempty(T.parameters) ? T : T.name.wrapper
end



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

function lpdf(d::Weibull, t)
    # parameterization of Lee and Wang (SAS)
    log(d.ρ) + log(d.γ) + (d.γ - 1.0) * log(d.ρ * t) - (d.ρ * t)^d.γ
end

function lsurv(d::Weibull, t)
    # parameterization of Lee and Wang (SAS)
    -(d.ρ * t)^d.γ
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
    log(d.ρ) - (d.ρ * t)
end

function lsurv(d::Exponential, t)
    # parameterization of Lee and Wang (SAS), survival uses Kalbfleisch and Prentice
    -d.ρ * t
end

shape(d::Exponential) = 1.0
scale(d::Exponential) = d.γ
params(d::Exponential) = (d.γ)

##################################################################################################################### 
# structs
#####################################################################################################################


mutable struct PSParms{
    D<:Matrix{<:Real},
    B<:Vector{<:Float64},
    R<:Vector{<:Float64},
    L<:Vector{<:Float64},
    G<:Union{Nothing,Vector{<:Float64}},
    H<:Union{Nothing,Matrix{<:Float64}},
    I<:Int,
} <: AbstractLSurvivalParms
    X::Union{Nothing,D}
    _B::B                        # coefficient vector
    _r::R                        # linear predictor/risk
    _LL::L                       # partial likelihood history
    _grad::G                     # gradient vector
    _hess::H                     # Hessian matrix
    n::I                         # number of observations
    p::I                         # number of parameters
end

function PSParms(X::Union{Nothing,D}; extraparms = 1) where {D<:AbstractMatrix}
    n, p = size(X)
    r = p + extraparms
    PSParms(
        X,
        fill(0.0, p),
        fill(0.0, n),
        zeros(Float64, 1),
        fill(0.0, r),
        fill(0.0, r, r),
        n,
        p,
    )
end

#Base.convert(PSParms, PHParms)

mutable struct PSModel{G<:LSurvivalResp,L<:AbstractLSurvivalParms,D<:AbstractSurvDist} <:
               AbstractPSModel
    R::Union{Nothing,G}        # Survival response
    P::L        # parameters
    formula::Union{FormulaTerm,Nothing}
    d::D
    fit::Bool
end
function PSModel(
    R::Union{Nothing,G},
    P::L,
    d::D,
) where {G<:LSurvivalResp,L<:AbstractLSurvivalParms,D<:AbstractSurvDist}
    PSModel(R, P, nothing, d, false)
end

function parsurvrisk!(_r, X, β)
    _r .= exp.(-X * β)
end

function parsurvrisk(X, β)
    exp.(-X * β)
end

function loglik(d::D, enter, exit, y, wts) where {D<:AbstractSurvDist}
    # ρ is linear predictor
    ll = enter > 0 ? lsurv(d, enter) : 0 # (anti)-contribution for all in risk set (cumulative conditional survival at entry)
    ll +=
        y == 1 ? lpdf(d, exit) : # extra contribution for events
        lsurv(d, exit) # extra contribution for censored (cumulative conditional survival at censoring)
    ll *= wts
    ll
end


function updatemu!(m::M, θ) where {M<:PSModel}
    m.P._B = θ[1:m.P.p]
    parsurvrisk!(m.P._r, m.P.X, m.P._B)
    remparms = setdiff(θ, θ[1:m.P.p])
    remparms
end

function getmu(m::M, θ) where {M<:PSModel}
    parsurvrisk(m.P.X, θ[1:m.P.p])
end



# theta includes linear predictor and other parameters
function ll_unfixedscale(m::M, θ) where {M<:PSModel}
    newidx = (m.P.p+1):length(θ)
    oldidx = 1:m.P.p
    # put theta into correct parameters
    #_r = getmu(m, θ)
    LL = 0.0
    for i = 1:length(m.R.enter)
        #https://stackoverflow.com/questions/70043313/get-simple-name-of-type-in-julia
        Distr = name(typeof(m.d))(exp(-sum(m.P.X[i, :] .* θ[oldidx])), exp.(θ[newidx])...)
        #d = Weibull(exp(-sum(m.P.X[i,:] .* θ[oldidx])), θ[newidx]...)
        LL += loglik(Distr, m.R.enter[i], m.R.exit[i], m.R.y[i], m.R.wts[i])
    end
    LL
end

function ll_fixedscale(m::M, θ) where {M<:PSModel}
    LL = 0.0
    for i = 1:length(m.R.enter)
        #https://stackoverflow.com/questions/70043313/get-simple-name-of-type-in-julia
        Distr = name(typeof(m.d))(exp(-sum(m.P.X[i, :] .* θ)))
        #d = Weibull(exp(-sum(m.P.X[i,:] .* θ[oldidx])), θ[newidx]...)
        LL += loglik(Distr, m.R.enter[i], m.R.exit[i], m.R.y[i], m.R.wts[i])
    end
    LL
end



function lgh!(m::M, _theta) where {M<:PSModel}
    #r = [updateparams(m.d, _theta) for i in eachindex(m.R.enter)]
    newidx = (m.P.p+1):length(θ)
    ll = newidx.start <= newidx.stop ? ll_unfixedscale : ll_fixedscale
    f(x) = ll(m, x)
    push!(m.P._LL, f(_theta))
    m.P._grad = gradient(f, _theta)[1]
    m.P._hess = Zygote.hessian(f, _theta)
    m.P._LL[end], m.P._grad, m.P._hess
end


"""

using LSurvival, Zygote
dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])
enter = zeros(length(dat1.time))
t = dat1.time
d = dat1.status
X = hcat(ones(length(dat1.x)), dat1.x)
wt = ones(length(t))
coxph(X[:,2:2],enter,t,d) # lnhr = 1.67686

include(expanduser("~/repo/LSurvival.jl/src/parsurvival.jl"))
include(expanduser("~/repo/LSurvival.jl/src/distributions.jl"))

dist = Weibull()
P = PSParms(X, extraparms=length(dist)-1)
R = LSurvivalResp(enter, t, d)    # specification with ID only
m = PSModel(R,P,dist)


θ = rand(length(m.P._B)+length(m.d)-1)

function _fit!(m;start=rand(length(m.P._B)+length(m.d)-1))
    parms = deepcopy(start)
    maxgrad = 1e12
    λ = 1.0
    while maxgrad > 1e-12
         lastgrad = deepcopy(maxgrad)
         lgh!(m, parms)
         maxgrad = m.P._grad'm.P._grad
         if lastgrad < maxgrad
             λ*=0.5
         else
            λ = min(λ*1.5, 1)
         end
         λ
         # newton raphson update
         parms .+= inv(-m.P._hess) * m.P._grad * λ
         λ
    end
    v = inv(-m.P._hess)
    parms, [sqrt(v[i,i]) for i in 1:size(v,2)]
end
p,se = _fit!(m)
"""