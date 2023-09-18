# Parametric survival models
######################################################################
# Survival model distributions
######################################################################

#abstract type AbstractSurvDist end

# Weibull
mutable struct Weibull{T<:Real} <: AbstractSurvDist
    ρ::T   # scale: linear effects on this parameter
    γ::T   # shape
end

function Weibull()
    Weibull(ones(Float64,2)...)
end

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

function updateparams!(d::Weibull, θ::Vector{T}) where {T<:Real}
    d.ρ = θ[1]
    d.γ = θ[2]
    d
end

function newdist(d::Weibull, θ::Vector{T}) where {T<:Real}
    typeof(d)(θ...)
end

mutable struct Exponential{T<:Real} <: AbstractSurvDist
    ρ::T   # scale (Weibull shape is 1.0)
end

function lpdf(d::Exponential, t)
    # parameterization of Lee and Wang (SAS)
    log(d.ρ) - (d.ρ * t)
end

function lsurv(d::Exponential, t)
    # parameterization of Lee and Wang (SAS)
    -d.ρ * t
end

shape(d::Exponential) = 1.0
scale(d::Exponential) = d.γ
params(d::Exponential) = (d.γ)
function updateparams!(d::Exponential, θ)
    d.ρ = θ[1]
end

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

function PSParms(X::Union{Nothing,D};extraparms=1) where {D<:AbstractMatrix}
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
    m.P._B = θ[1:m.P.p]
    _r = parsurvrisk(m.P.X, m.P._B)
    remparms = setdiff(θ, θ[1:m.P.p])
    remparms, _r
end


function updatedist!(m::M, parms) where {M<:PSModel}

    remparms
end



# theta includes linear predictor and other parameters
function ll!(m::M, θ) where {M<:PSModel}
    # put theta into correct parameters
    η = updatemu!(m, θ)
    disti = [updateparams!(m.d, vcat(m.P._r[i], η)) for i in eachindex(m.R.enter)]
    LL = 0.0
    for i in eachindex(m.R.enter)
        LL +=
            loglik(disti[i], m.R.enter[i], m.R.exit[i], m.R.y[i], m.R.wts[i])
    end
    LL
end

function ll(m::M, θ) where {M<:PSModel}
    # put theta into correct parameters
    m.P._B = θ[1:m.P.p]
    η, _r = getmu(m, θ)
    disti = [newdist(m.d, vcat(_r[i], η)) for i in eachindex(m.R.enter)]
    LL = 0.0
    for i in eachindex(m.R.enter)
        LL +=
            loglik(disti[i], m.R.enter[i], m.R.exit[i], m.R.y[i], m.R.wts[i])
    end
    LL
end


function lgh!(m::M, _theta, x, enter, exit, d, wt) where {M<:PSModel}
    #r = [updateparams(m.d, _theta) for i in eachindex(m.R.enter)]
    llt(_theta) = ll(m, _theta)
    _loglik = llt(_theta)
    
    gradient( parm -> ll(m, parm), _theta)[1]

    ReverseDiff.gradient(llt, _theta)
    h = ForwardDiff.hessian!(m.P._hess, parm -> ll(m, parm), _theta)
    _loglik, g, h
end


"""
# ForwardDiff is algorithmically more efficient for differentiating functions where the input dimension 
# is less than the output dimension, while ReverseDiff is algorithmically more efficient for differentiating 
# functions where the output dimension is less than the input dimension

#using ReverseDiff
using LSurvival
dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])
enter = zeros(length(dat1.time))
t = dat1.time
d = dat1.status
X = hcat(ones(length(dat1.x)), dat1.x)
wt = ones(length(t))
coxph(X[:,2:2],enter,t,d) # lnhr = 1.67686

include(expanduser("~/repo/LSurvival.jl/src/parsurvival.jl"))
include(expanduser("~/repo/LSurvival.jl/src/distributions.jl"))


P = PSParms(X, extraparms=1)
R = LSurvivalResp(enter, t, d)    # specification with ID only

m = PSModel(R,P,Weibull())

θ = rand(size(X,2)+1)
_theta = θ

function fitexpon()
    parms = rand(2)
    _ll, _grad, _hess = lgh!(parms, x, enter, t, d, wt; ll = lle)
    maxgrad = _grad'_grad
    while maxgrad > 1e-12
    #for j in 1:10
        _ll, _grad, _hess = lgh!(parms, x, enter, t, d, wt; ll = lle)
        # newton raphson update
        parms .+= inv(-_hess) * _grad
        maxgrad = _grad'_grad
    end
    v = inv(-_hess)
    parms, [sqrt(v[i,i]) for i in 1:size(v,2)]
end
p,se = fitexpon()
"""