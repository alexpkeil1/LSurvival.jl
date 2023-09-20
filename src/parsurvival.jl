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
    S<:Vector{<:Float64},
    I<:Int,
} <: AbstractLSurvivalParms
    X::Union{Nothing,D}
    _B::B                        # coefficient vector
    _r::R                        # linear predictor/risk
    _LL::L                       # partial likelihood history
    _grad::G                     # gradient vector
    _hess::H                     # Hessian matrix
    _S::S                         # Scale parameter(s)
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
        zeros(extraparms),
        n,
        p,
    )
end


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
    np = length(d)
    P._S = zeros(np-1)
    r = P.p + np-1
    P._grad = fill(0.0, r)
    P._hess = fill(0.0, r, r)
    PSModel(R, P, nothing, d, false)
end

params(m::PSModel) = vcat(m.P._B, m.P._S)


"""
Log likelihood contribution for an observation in a parametric survival model
"""
function loglik(d::D, enter, exit, y, wts) where {D<:AbstractSurvDist}
    # ρ is linear predictor
    ll = enter > 0 ? -lsurv(d, enter) : 0 # (anti)-contribution for all in risk set (cumulative conditional survival at entry)
    ll +=
        y == 1 ? lpdf(d, exit) : # extra contribution for events plus the log of the Jacobian of the transform on time
        lsurv(d, exit) # extra contribution for censored (cumulative conditional survival at censoring)
    ll *= wts
    ll
end




# theta includes linear predictor and other parameters
"""
i=1
θ = rand(3)
Distr = name(typeof(m.d))(dot(m.P.X[i:i, :], θ[oldidx]), θ[newidx][1])              # scale, linear model
LSurvival.lpdf(Distr, m.R.exit[i])
t = m.R.exit[i]
"""
function ll_unfixedscale(m::M, θ) where {M<:PSModel}
    newidx = (m.P.p+1):length(θ)
    oldidx = 1:m.P.p
    # put theta into correct parameters
    LL = 0.0
    for i = 1:length(m.R.enter)
        #https://stackoverflow.com/questions/70043313/get-simple-name-of-type-in-julia
        #Distr = name(typeof(m.d))(exp(-sum(m.P.X[i, :] .* θ[oldidx])), exp.(θ[newidx])...) # log scale, exponential mean model
        #Distr = name(typeof(m.d))(exp(-sum(m.P.X[i, :] .* θ[oldidx])), θ[newidx]...)       # scale, exponential mean model
        #Distr = name(typeof(m.d))(dot(m.P.X[i:i, :], θ[oldidx]), θ[newidx.start])              # scale, linear model
        Distr = name(typeof(m.d))(dot(m.P.X[i:i, :], θ[oldidx]), exp(θ[newidx.start]))              # log scale, linear model
        #d = Weibull(exp(-sum(m.P.X[i,:] .* θ[oldidx])), θ[newidx]...)
        LL += loglik(Distr, m.R.enter[i], m.R.exit[i], m.R.y[i], m.R.wts[i])
    end
    LL
end


function ll_fixedscale(m::M, θ) where {M<:PSModel}
    LL = 0.0
    for i = 1:length(m.R.enter)
        #https://stackoverflow.com/questions/70043313/get-simple-name-of-type-in-julia
        #Distr = name(typeof(m.d))(exp(-sum(m.P.X[i, :] .* θ)))
        Distr = name(typeof(m.d))(dot(m.P.X[i:i, :], θ))              # log scale, linear model
        #d = Weibull(exp(-sum(m.P.X[i,:] .* θ[oldidx])), θ[newidx]...)
        LL += loglik(Distr, m.R.enter[i], m.R.exit[i], m.R.y[i], m.R.wts[i])
    end
    LL
end

"""
i=1
θ = [2.23733, -0.744225, 0.456316]
_theta = θ
f(_theta)
gradient(f, _theta)
Zygote.hessian(f, _theta)
Zygote.hessian_reverse(f, _theta)
y, back = Zygote.pullback(f, _theta)
back(_theta)


"""
function lgh!(m::M, _theta) where {M<:PSModel}
    #r = [updateparams(m.d, _theta) for i in eachindex(m.R.enter)]
    newidx = (m.P.p+1):length(_theta)
    ll = newidx.start <= newidx.stop ? ll_unfixedscale : ll_fixedscale
    f(x) = ll(m, x)
    push!(m.P._LL, f(_theta))
    m.P._grad = gradient(f, _theta)[1]
    m.P._hess = Zygote.hessian(f, _theta)
    m.P._LL[end], m.P._grad, m.P._hess
end

function setinits(m::M) where {M<:PSModel}
    startint = m.P.X\log.(m.R.exit)
    startscale = std(log.(m.R.exit) .- m.P.X*startint)
    start0 = vcat(startint, log(startscale))
    start0[1:length(params(m))]
end


"""
using LSurvival
dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])

X = ones(length(dat1.x),1)
dist = LSurvival.Weibull()
P = PSParms(X, extraparms=length(dist)-1)
P._B
P._grad
R = LSurvivalResp(dat1.time, dat1.status)    # specification with ID only
m = PSModel(R,P,dist)
m.P._grad

#parms = vcat(zeros(length(m.P._grad)-1), 1.0)
parms = [2.001, .551]
_theta = parms
θ = _theta


"""
function _fit!(
    m::PSModel;
    verbose::Bool = false,
    maxiter::Integer = 500,
    gtol::Float64 = 1e-8,
    start = nothing,
    keepx = true,
    keepy = true,
    bootstrap_sample = false,
    bootstrap_rng = MersenneTwister(),
    kwargs...,
)
    m = bootstrap_sample ? bootstrap(bootstrap_rng, m) : m
    start = !isnothing(start) ? start : setinits(m)
    parms = deepcopy(start)
    λ = 1.0
    #
    totiter = 0
    #lgh!(m, parms)
    m.P._grad .+= 100.0
    oldQ = floatmax()
    while totiter < maxiter
        totiter += 1
        converged = (maximum(abs.(m.P._grad)) < gtol)
        if converged
            break
        end
        lgh!(m, parms)
        Q = m.P._grad' * m.P._grad #l2 norm of vector
        if Q > oldQ # gradient has increased, indicating the maximum  was overshot
            λ *= 0.5  # step-halving
        else
            λ = min(2.0λ, 1.0) # de-halving
        end
        isnan(m.P._LL[end]) ?
        throw("Log-likelihood is NaN: try different starting values") : true
        if abs(m.P._LL[end]) != Inf
            parms .+= inv(-m.P._hess) * m.P._grad * λ
            oldQ = Q
        else
            @debug "Log-likelihood history: $_llhistory $(m.P._LL[1])"
            throw("Log-likelihood is not finite: check model inputs")
        end
        # newton raphson update
        verbose ? println(m.P._LL[end]) : true
    end
    if (totiter == maxiter) && (maxiter > 0)
        @warn "Algorithm did not converge after $totiter iterations: check for collinearity of predictors"
        @debug "recent log-likelihood history: $(_llhistory[end-max(10,maxiter-1):end]) $(m.P._LL[1])"
    end
    if verbose && (maxiter == 0)
        @warn "maxiter = 0, model coefficients set to starting values"
    end
    m.fit = true
    m.P._LL = m.P._LL[2:end]
    m.P._B .= parms[1:m.P.p]
    m.P._S .= parms[(m.P.p+1):end]
    m.P.X = keepx ? m.P.X : nothing
    m.R = keepy ? m.R : nothing
    m
end

function StatsBase.fit!(
    m::PSModel;
    verbose::Bool = false,
    maxiter::Integer = 500,
    gtol::Float64 = 1e-8,
    start = nothing,
    kwargs...,
)
    if haskey(kwargs, :maxIter)
        Base.depwarn("'maxIter' argument is deprecated, use 'maxiter' instead", :fit!)
        maxiter = kwargs[:maxIter]
    end
    if haskey(kwargs, :convTol)
        Base.depwarn("'convTol' argument is deprecated, use `gtol` instead", :fit!)
        gtol = kwargs[:convTol]
    end
    if !issubset(keys(kwargs), (:maxIter, :convTol, :tol, :keepx, :keepy, :getbasehaz))
        throw(ArgumentError("unsupported keyword argument in: $(kwargs...)"))
    end
    if haskey(kwargs, :tol)
        Base.depwarn("`tol` argument is deprecated, use `gtol` instead", :fit!)
        gtol = kwargs[:tol]
    end

    #start = isnothing(start) ? setinits(m) : start

    _fit!(m, verbose = verbose, maxiter = maxiter, gtol = gtol, start = start; kwargs...)
end


"""
R = LSurvivalResp(enter, exit, y, wts, id)
P0 = PSParms(ones(size(X,1),1), extraparms = length(dist) - 1)
#M = PSModel
res0 = M(R, P0, dist)
start0 = LSurvival.setinits(res0)
start0 = [2.00124, log(0.550543)]

fit!(res0, start=start0, maxiter=1);
res0.P._LL
res0.P._grad
res0.P._hess
res0.P._B

function grad_wlp(ρ, γ, t)  
    [(exp((log(t) - ρ) / γ) - 1.0) / γ,
    (ρ + log(t)*exp((log(t) - ρ) / γ) - γ - log(t) - ρ*exp((log(t) - ρ) / γ)) / (γ^2)]
end
function grad_wls(ρ, γ, t)
    [exp((log(t) - ρ) / γ) / γ,
    ((log(t) - ρ)*exp((log(t) - ρ) / γ)) / (γ^2)]
end

ρ = res0.P._B[1]
γ = exp(res0.P._S[1])
t = R.exit
y = R.y
explicit_gradient = grad_w(1, 1, 1) .* 0.0  
for (i,t) in enumerate(R.exit)
    explicit_gradient .+= y[i] > 0 ? grad_wlp(ρ, γ, t) :  grad_wls(ρ, γ, t)  
end
explicit_gradient
res0.P._grad
res0.P._hess

"""
function fit(
    ::Type{M},
    X::Matrix{<:Real},#{<:FP},
    enter::Vector{<:Real},
    exit::Vector{<:Real},
    y::Y;
    dist = Weibull(),
    id::Vector{<:AbstractLSurvivalID} = [ID(i) for i in eachindex(y)],
    wts::Vector{<:Real} = similar(enter, 0),
    offset::Vector{<:Real} = similar(enter, 0),
    fitargs...,
) where {M<:PSModel,Y<:Union{Vector{<:Real},BitVector}}

    # Check that X and y have the same number of observations
    if size(X, 1) != size(y, 1)
        throw(DimensionMismatch("number of rows in X and y must match"))
    end
        
    R = LSurvivalResp(enter, exit, y, wts, id)
    P0 = PSParms(ones(size(X,1),1), extraparms = length(dist) - 1)
    res0 = M(R, P0, dist)
    start0 = LSurvival.setinits(res0)
    fit!(res0, start=start0, maxiter=1);
    #

    P = PSParms(X, extraparms = length(dist) - 1)
    if !haskey(fitargs, :start)
        st=zeros(length(P._grad))
        st[1] = params(res0)[1]
        st[end] = params(res0)[end]
        fitargs = (start=st, fitargs...)
    end
    res = M(R, P, dist)
    push!(res.P._LL, res0.P._LL[end])
    return fit!(res; fitargs...)
end

function modelframe(
    f::FormulaTerm,
    data,
    contrasts::AbstractDict,
    ::Type{M},
) where {M<:PSModel}
    # closely adapted from GLM.jl
    Tables.istable(data) ||
        throw(ArgumentError("expected data in a Table, got $(typeof(data))"))
    t = Tables.columntable(data)
    msg = StatsModels.checknamesexist(f, t)
    msg != "" && throw(ArgumentError(msg))
    data, _ = StatsModels.missing_omit(t, f)
    sch = schema(f, data, contrasts)
    f = apply_schema(f, sch, M)
    f, modelcols(f, data)
end


function fit(
    ::Type{M},
    f::FormulaTerm,
    data;
    dist = Weibull(),
    id::AbstractVector{<:AbstractLSurvivalID} = [
        ID(i) for i in eachindex(getindex(data, 1))
    ],
    wts::AbstractVector{<:Real} = similar(getindex(data, 1), 0),
    offset::AbstractVector{<:Real} = similar(getindex(data, 1), 0),
    contrasts::AbstractDict{Symbol} = Dict{Symbol,Any}(),
    fitargs...,
) where {M<:PSModel}
    f, (y, X) = modelframe(f, data, contrasts, M)

    R = LSurvivalResp(y, wts, id)

    P0 = PSParms(ones(size(X,1),1), extraparms = length(dist) - 1)
    res0 = M(R, P0, dist)
    start0 = LSurvival.setinits(res0)
    fit!(res0, start=start0)

    P = PSParms(X, extraparms = length(dist) - 1)
    if !haskey(fitargs, :start)
        st=zeros(length(P._grad))
        st[1] = params(res0)[1]
        st[end] = params(res0)[end]
        fitargs = (start=st, fitargs...)
    end
    res = M(R, P, f, dist, false)

    push!(res.P._LL, res0.P._LL[end])
    return fit!(res; fitargs...)
end

survreg(X, enter, exit, y, args...; kwargs...) =
    fit(PSModel, X, enter, exit, y, args...; kwargs...)

survreg(f::FormulaTerm, data; kwargs...) = fit(PSModel, f, data; kwargs...)

##################################################################################################################### 
# summary functions for PSModel objects
#####################################################################################################################

formula(x::M) where {M<:PSModel} = x.formula
scale(x::M) where {M<:PSModel} = x.P._S

StatsBase.coefnames(x::M) where {M<:PSModel} =
    x.formula === nothing ? ["b$i" for i = 1:length(coef(x))] : coefnames(formula(x).rhs)

function StatsBase.coef(m::M) where {M<:PSModel}
    mwarn(m)
    m.P._B
end

# quantile function for standard normal

function StatsBase.coeftable(m::M; level::Float64 = 0.95) where {M<:PSModel}
    mwarn(m)
    β = vcat(coef(m), scale(m))
    std_err = stderror(m)
    #zcrit = quantile.(Normal(), [(1 - level) / 2, 1 - (1 - level) / 2])
    zcrit = qstdnorm.([(1 - level) / 2, 1 - (1 - level) / 2])
    lci = β .+ zcrit[1] * std_err
    uci = β .+ zcrit[2] * std_err
    z = β ./ std_err
    pval = calcp.(z)
    op = hcat(β, std_err, lci, uci, z, pval)
    head = ["Est", "StdErr", "LCI", "UCI", "Z", "P(>|Z|)"]
    pcol = 6
    zcol = 5
    #rown = ["b$i" for i = 1:size(op)[1]]
    rown = vcat(coefnames(m), ["log(Scale)" for i = 1:length(scale(m))])
    rown = typeof(rown) <: AbstractVector ? rown : [rown]
    if length(m.P._grad) > length(m.P._B)
        #println("Scale parameter")
        true
    end
    StatsBase.CoefTable(op, head, rown, pcol, zcol)
end



function StatsBase.confint(m::M; level::Float64 = 0.95, kwargs...) where {M<:PSModel}
    mwarn(m)
    beta = coef(m)
    std_err = stderror(m; kwargs...) # can have type="robust"
    z = beta ./ std_err
    #zcrit = quantile.(Distributions.Normal(), [(1 - level) / 2, 1 - (1 - level) / 2])
    zcrit = qstdnorm.([(1 - level) / 2, 1 - (1 - level) / 2])
    lci = beta .+ zcrit[1] * std_err
    uci = beta .+ zcrit[2] * std_err
    hcat(lci, uci)
end

function StatsBase.fitted(m::M) where {M<:PSModel}
    mwarn(m)
    D = modelmatrix(m)
    D * coef(m)
end

function StatsBase.isfitted(m::M) where {M<:PSModel}
    m.fit
end

function StatsBase.deviance(m::M) where {M<:PSModel}
    mwarn(m)
    -2 * loglikelihood(m)
end

function StatsBase.aic(m::M) where {M<:PSModel}
    deviance(m) + 2 * dof(m)
end

function StatsBase.aicc(m::M) where {M<:PSModel}
    df = dof(m)
    aic(m) + 2 * df * (df - 1) / (dof_residual(m) - 1)
end

function StatsBase.bic(m::M) where {M<:PSModel}
    deviance(m) + dof(m) * log(nobs(m))
end


function StatsBase.nulldeviance(m::M) where {M<:PSModel}
    mwarn(m)
    -2 * nullloglikelihood(m)
end

function StatsBase.dof(m::M) where {M<:PSModel}
    mwarn(m)
    m.P.p
end

function StatsBase.dof_residual(m::M) where {M<:PSModel}
    mwarn(m)
    nobs(m) - dof(m)
end


function StatsBase.nobs(m::M) where {M<:PSModel}
    mwarn(m)
    length(unique(m.R.id))
end

"""
Maximum log likelihood for a fitted `PSModel` model
"""
function StatsBase.loglikelihood(m::M) where {M<:PSModel}
    mwarn(m)
    m.P._LL[end]
end

function StatsBase.modelmatrix(m::M) where {M<:PSModel}
    mwarn(m)
    m.P.X
end

"""
Null log-partial likelihood for a fitted `PSModel` model

Note: this is just the log partial likelihood at the initial values of the model, which default to 0. If initial values are non-null, then this function no longer validly returns the null log-partial likelihood.
"""
function StatsBase.nullloglikelihood(m::M) where {M<:PSModel}
    mwarn(m)
    m.P._LL[1]
end

function StatsBase.model_response(m::M) where {M<:PSModel}
    mwarn(m)
    m.R
end

StatsBase.response(m::M) where {M<:PSModel} = StatsBase.model_response(m)

function StatsBase.score(m::M) where {M<:PSModel}
    mwarn(m)
    m.P._grad
end

function StatsBase.stderror(m::M; kwargs...) where {M<:PSModel}
    mwarn(m)
    sqrt.(diag(vcov(m; kwargs...)))
end

function StatsBase.vcov(m::M; type::Union{String,Nothing} = nothing) where {M<:PSModel}
    mwarn(m)
    if type == "robust"
        #res = robust_vcov(m)
        throw("$type not implemented")
    elseif type == "jackknife"
        #res = jackknife_vcov(m)
        throw("$type not implemented")
    else
        res = -inv(m.P._hess)
    end
    res
end

function StatsBase.weights(m::M) where {M<:PSModel}
    mwarn(m)
    m.R.wts
end


function Base.show(io::IO, m::M; level::Float64 = 0.95) where {M<:PSModel}
    if !m.fit
        println(io, "Model not yet fitted")
        return nothing
    end
    ll = loglikelihood(m)
    llnull = nullloglikelihood(m)
    chi2 = 2 * (ll - llnull)
    coeftab = coeftable(m, level = level)
    df = length(coeftab.rownms)-1
    #lrtp = 1 - cdf(Distributions.Chisq(df), chi2)
    #lrtp = 1 - cdf(Chisq(df), chi2)
    lrtp = 1 - cdfchisq(df, chi2)
    iob = IOBuffer()
    println(iob, coeftab)
    str = """\nMaximum likelihood estimates (alpha=$(@sprintf("%.2g", 1-level))):\n"""
    str *= String(take!(iob))
    str *= "Log-likelihood (Intercept only): $(@sprintf("%8g", llnull))\n"
    str *= "Log-likelihood (full): $(@sprintf("%8g", ll))\n"
    str *= "LRT p-value (X^2=$(round(chi2, digits=2)), df=$df): $(@sprintf("%.5g", lrtp))\n"
    str *= "Newton-Raphson iterations: $(length(m.P._LL)-1)"
    println(io, str)
end

Base.show(m::M; kwargs...) where {M<:PSModel} = Base.show(stdout, m; kwargs...)


"""

using LSurvival, Zygote, Random, StatsBase, Printf, Tables
dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])
enter = zeros(length(dat1.time))
t = dat1.time
d = dat1.status
X = hcat(ones(length(dat1.x)), dat1.x)
wt = ones(length(t))
coxph(X[:,2:2],enter,t,d) # lnhr = 1.67686
res = survreg(@formula(Surv(time,status)~x), dat1, dist=LSurvival.Exponential())
res = survreg(@formula(Surv(time,status)~x), dat1, dist=LSurvival.Weibull());
res = survreg(@formula(Surv(time,status)~x), dat1, dist=LSurvival.Weibull(), start = [2.4, -1, 0]);


#include(expanduser("~/repo/LSurvival.jl/src/parsurvival.jl"))
#include(expanduser("~/repo/LSurvival.jl/src/distributions.jl"))

dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])

dist = LSurvival.Weibull()
P = PSParms(X[:,1:1], extraparms=length(dist)-1)
P = PSParms(X, extraparms=length(dist)-1)
P._B
P._grad
R = LSurvivalResp(dat1.time, dat1.status)    # specification with ID only
m = PSModel(R,P,dist)

m.P._B
m.P._S
m.P._LL

LSurvival._fit!(m);
dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])
survreg(@formula(Surv(time,status)~x), dat1, dist=LSurvival.Weibull())
"""

;
