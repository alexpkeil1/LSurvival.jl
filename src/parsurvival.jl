# Parametric survival models
######################################################################
# Convenience functions
######################################################################

#abstract type AbstractSurvDist end
#=
function copydist(d::T, args...) where {T<:AbstractSurvDist}
    typeof(d)(args...)
end
=#
function name(::Type{T}) where {T}
    #https://stackoverflow.com/questions/70043313/get-simple-name-of-type-in-julia
    isempty(T.parameters) ? T : T.name.wrapper
end


function Base.length(d::T) where {T<:AbstractSurvDist}
    length(fieldnames(T))
end




##################################################################################################################### 
# model related structs
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
    _S::S                         # log(Scale) parameter(s)
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
        Float64[],
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
    P._S = zeros(np - 1) # Gen gamma, this will have length 2, Expon: 0
    r = P.p + np - 1
    P._grad::Vector{Float64} = fill(0.0, r)
    P._hess::Matrix{Float64} = fill(0.0, r, r)
    PSModel(R, P, nothing, d, false)
end


##################################################################################################################### 
# Likelihood functions
#####################################################################################################################


"""
Log likelihood contribution for an observation in a parametric survival model


 ```julia
    d = m.d
    i = 1
    enter = m.R.enter[i]
    exit = m.R.exit[i]
    y = m.R.y[i]
    wts = m.R.wts[i]
    x = m.P.X[i,:]
    θ=[1,0,.4]
    
    m.P._B

```
"""
function loglik(d::D, θ, enter, exit, y, x, wts) where {D<:AbstractSurvDist}
    # left truncation
    ll = enter > 0 ? -lsurv(d, θ, enter, x) : 0 # (anti)-contribution for all in risk set (cumulative conditional survival at entry)
    # event, right censoring
    ll +=
        y > 0 ? lpdf(d, θ, exit, x) : # extra contribution for events plus the log of the Jacobian of the transform on time
        lsurv(d, θ, exit, x) # extra contribution for censored (cumulative conditional survival at censoring)
    # interval censoring
    # note this will add back the initial reduction from late entry if enter > 0
    #=
    # note: this needs to be integrated with the event/right censoring column
    #   as written, it will cancel out the right censoring part
    if y < 0
        ll +=  -lsurv(d, θ, exit, x) + (
            enter > 0 ? lsurv(d, θ, enter, x) : 1
            )
    end
    =# 
    ll *= wts
    ll
end

"""
Gradient contribution for an observation in a parametric survival model


 ```julia
    d = m.d
    i = 1
    enter = m.R.enter[i]
    exit = m.R.exit[i]
    y = m.R.y[i]
    wts = m.R.wts[i]
    x = m.P.X[i,:]
    θ=[1,0,.4]

```
"""
function dloglik!(gt, d::D, θ, enter, exit, y, x, wts) where {D<:AbstractSurvDist}
    gt .= enter > 0 ? -lsurv_gradient(d, θ, enter, x) : gt .* 0.0 # (anti)-contribution for all in risk set (cumulative conditional survival at entry)
    gt .+=
        y > 0 ? lpdf_gradient(d, θ, exit, x) : # extra contribution for events plus the log of the Jacobian of the transform on time
        lsurv_gradient(d, θ, exit, x) # extra contribution for censored (cumulative conditional survival at censoring)
    #=    left censoring
    # note: this needs to be integrated with the event/right censoring column
    #   as written, it will cancel out the right censoring part

    gt .+=
        y < 0 ? -lsurv_gradient(d, θ, exit, x) + (
            enter > 0 ? lsurv_gradient(d, θ, enter, x) : 1
            )        
    =#
    gt .*= wts
    gt
end

"""
Hessian contribution for an observation in a parametric survival model


 ```julia
    d = m.d
    i = 1
    enter = m.R.enter[i]
    exit = m.R.exit[i]
    y = m.R.y[i]
    wts = m.R.wts[i]
    x = m.P.X[i,:]
    θ=[1,0,.4]

```
"""
function ddloglik!(he, d::D, θ, enter, exit, y, x, wts) where {D<:AbstractSurvDist}
    he .= enter > 0 ? -lsurv_hessian(d, θ, enter, x) : he .* 0.0 # (anti)-contribution for all in risk set (cumulative conditional survival at entry)
    he .+=
        y > 0 ? lpdf_hessian(d, θ, exit, x) : # extra contribution for events plus the log of the Jacobian of the transform on time
        lsurv_hessian(d, θ, exit, x) # extra contribution for censored (cumulative conditional survival at censoring)
    #=    left censoring
    # note: this needs to be integrated with the event/right censoring column
    #   as written, it will cancel out the right censoring part

    he .+=
        y < 0 ? -lsurv_hessian(d, θ, exit, x) + (
            enter > 0 ? lsurv_hessian(d, θ, enter, x) : 1
            )        
    =#
    he .*= wts
    he
end



#lpdf(d::Weibull, θ, t, x)


# theta includes linear predictor and other parameters
function get_ll(m::M, θ) where {M<:PSModel}
    LL = 0.0
    for i = 1:length(m.R.enter)
        LL += loglik(m.d, θ, m.R.enter[i], m.R.exit[i], m.R.y[i], m.P.X[i, :], m.R.wts[i])
    end
    LL
end


function get_gradient(m::M, θ) where {M<:PSModel}
    grad = zeros(length(θ))
    gt = zeros(length(θ))
    for i = 1:length(m.R.enter)
        gt .*= 0.0
        dloglik!(gt, m.d, θ, m.R.enter[i], m.R.exit[i], m.R.y[i], m.P.X[i, :], m.R.wts[i])
        grad += gt
    end
    grad
end


function get_hessian(m::M, θ) where {M<:PSModel}
    hess = zeros(length(θ), length(θ))
    he = zeros(length(θ), length(θ))
    for i = 1:length(m.R.enter)
        he .*= 0.0
        ddloglik!(he, m.d, θ, m.R.enter[i], m.R.exit[i], m.R.y[i], m.P.X[i, :], m.R.wts[i])
        hess += he
    end
    hess
end


"""
dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])
enter = zeros(length(dat1.time))
t = dat1.time
d = dat1.status
X = hcat(ones(length(dat1.x)), dat1.x)
wt = ones(length(t))

dist = Exponential()
P = PSParms(X[:,1:1], extraparms=length(dist)-1)
P = PSParms(X, extraparms=length(dist)-1)
P._B
P._grad
R = LSurvivalResp(dat1.time, dat1.status)    # specification with ID only
m = PSModel(R,P,dist)

λ=1
θ = rand(2)
lgh!(m, θ)
θ .+= inv(-m.P._hess) * m.P._grad * λ

lgh!(m, θ .+ [-.00, 0, 0])
lgh!(m, θ .+ [-.01, 0.05, 0.05])

m.P._LL



"""
function lgh!(m::M, θ) where {M<:PSModel}
    #r = [updateparams(m.d, θ) for i in eachindex(m.R.enter)]
    push!(m.P._LL, get_ll(m, θ))
    m.P._grad .= get_gradient(m, θ)
    m.P._hess .= get_hessian(m, θ)
    m.P._LL[end], m.P._grad, m.P._hess
end

function lonly!(m::M, θ) where {M<:PSModel}
    #r = [updateparams(m.d, θ) for i in eachindex(m.R.enter)]
    push!(m.P._LL, get_ll(m, θ))
    m.P._LL[end]
end


function setinits(m::M; verbose = false) where {M<:PSModel}
    # This function is unfortunately crucial
    #
    # Commented text from the "survival" package source code:
    # A good initial value of the scale turns out to be critical for successful
    #   iteration, in a surprisingly large number of data sets.
    # The best way we've found to get one is to fit a model with only the
    #   mean and the scale.  We don't need to do this in 3 situations:
    #    1. The only covariate is a mean (this step is then just a duplicate
    #       of the main fit).
    #    2. There are no scale parameters to estimate
    #    3. The user gave initial estimates for the scale
    # However, for 2 and 3 we still want the loglik for a mean only model
    #  as a part of the returned object.
    ly = log.(m.R.exit) #.+ (1.0 .- m.R.y) .* 0.5*median(log.(m.R.exit))
    startint = m.P.X \ ly
    startscale = std(ly) * sqrt(pi^2 / 6.0)
    # Commented text from the "survival" package source code:
    # We sometimes get into trouble with a small estimate of sigma,
    #  (the surface isn't SPD), but never with a large one.  Double it.
    # my comment: final starting points refer to exp(κ) where that equals 1 in the limit of the Weibull model, 
    # but it seems to do better with values closer to zero, which places it somewhere closer to the exponential
    start0 = vcat(startint, log(startscale * 2.0) / 2.0, -10.0 .* ones(2))[1:length(params(m))]
    verbose && println("Starting values: $(start0)")
    start0
end


##################################################################################################################### 
# Fitting functions
#####################################################################################################################

#"""
#```julia
#using LSurvival
#dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])
#
#X = ones(length(dat1.x),1)
#dist = LSurvival.Weibull()
#P = PSParms(X, extraparms=length(dist)-1)
#P._B
#P._grad
#R = LSurvivalResp(dat1.time, dat1.status)    # specification with ID only
#m = PSModel(R,P,dist)
#m.P._grad
#
##parms = vcat(zeros(length(m.P._grad)-1), 1.0)
#parms = [2.001, .551]
#θ = parms
#θ = θ
#```
#
#"""

#=
# Newton raphson algorithm for parametric survival models depending on analytic gradients and Hessians
# Never got this to work for some reason - the steps were always way too big for any model with a scale parameter
# but Hessian and gradient never failed any checks
function old_fit!(
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
    start = !isnothing(start) ? start : setinits(m, verbose=verbose)
    parms = deepcopy(start)
    λ = 1.0
    totiter = 0
    oldQ = floatmax()
    lgh!(m, parms)
    while totiter < maxiter
        totiter += 1
        converged = (maximum(abs.(m.P._grad)) < gtol)
        if converged
            break
        end
        verbose && println("$parms $(m.P._grad) $(m.P._hess)")
        verbose && println("$(m.P._LL[end])")
        Q = m.P._grad' * m.P._grad #l2 norm of vector
        if Q > oldQ # gradient has increased, indicating the maximum  was overshot
            stepfacmin = 0.001
            λ = max(λ*0.5, stepfacmin)  # step-halving
        else
            λ = min(2.0λ, 1.0) # de-halving
        end
        isnan(m.P._LL[end]) ?
        throw("Log-likelihood is NaN: try different starting values") : true
        if abs(m.P._LL[end]) != Inf
            parms .-= inv(m.P._hess) * m.P._grad * λ
            oldQ = Q
        else
            @debug "Log-likelihood history: $_llhistory $(m.P._LL[1])"
            throw("Log-likelihood is not finite: check model inputs")
        end
        lgh!(m, parms)
        # newton raphson update
        #verbose && println(m.P._LL[end])
    end
    if (totiter == maxiter) && (maxiter > 0)
        @warn "Algorithm did not converge after $totiter iterations: check for collinearity of predictors"
        @debug "recent log-likelihood history: $(_llhistory[end-max(10,maxiter-1):end]) $(m.P._LL[1])"
    end
    if verbose && (maxiter == 0)
        @warn "maxiter = 0, model coefficients set to starting values"
    end
    m.fit = true
    m.P._LL = m.P._LL
    m.P._B .= parms[1:m.P.p]
    m.P._S .= parms[(m.P.p+1):end]
    m.P.X = keepx ? m.P.X : nothing
    m.R = keepy ? m.R : nothing
    m
end
=#

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
    #start = nothing
    # verbose= false
    start = !isnothing(start) ? start : setinits(m, verbose = verbose)
    parms = deepcopy(start)

    nohess = issubset(["$(nameof(typeof(m.d)))"], ["GGamma", "Gamma"])
    updatefun! = nohess ? lonly! : lgh!

    function parmupdate!(F, G, H, θ, m;)
        m.P._LL[end] = isnothing(F) ? m.P._LL[end] : F
        m.P._grad = isnothing(G) ? m.P._grad : G
        m.P._hess = isnothing(H) ? m.P._hess : H
        #
        updatefun!(m, θ)
        # turn into a minimization problem
        m.P._B .= θ[1:m.P.p]
        m.P._S .= θ[(m.P.p+1):end]
        F = -m.P._LL[end]
        m.P._grad .*= -1.0
        m.P._hess .*= -1.0
        F
    end

    if nohess
        push!(m.P._LL, -10.0) # gets over-written
        ff(beta) = parmupdate!(nothing, nothing, nothing, beta, m)
        fgh! = TwiceDifferentiable(
            ff,
            parms
        )
    else
        fgh! = TwiceDifferentiable(
            only_fgh!((F, G, H, beta) -> parmupdate!(F, G, H, beta, m)),
            parms,
        )
    end
    updatefun!(m, parms)
    res = optimize(
        fgh!,
        parms,
        BFGS(
            linesearch = nohess ? LineSearches.BackTracking(order=2) :  LineSearches.HagerZhang(),
            alphaguess = nohess ? LineSearches.InitialQuadratic() : LineSearches.InitialStatic(scaled=false)
        ),
        Options(
            store_trace = true,
            iterations = maxiter,
            g_tol = gtol,
            show_trace = verbose,
            extended_trace = true,
        ),
    )
    m.P._grad .*= -1.0
    m.P._hess .*= -1.0
    if nohess
        verbose && println(res)
        G = -res.trace[end].metadata["g(x)"]
        H = inv(-res.trace[end].metadata["~inv(H)"])
        verbose && println("Using BFGS estimate of Hessian $H")
        m.P._hess .= H
        m.P._grad = G
    end
    maxiter == 0 && @warn("maxiter=0: no fitting done")
    !converged(res) &&  # this line has been touchy!
        maxiter > 0 &&
        @warn("Optimizer reports model did not converge. Gradient: $(m.P._grad)")

    m.fit = true
    m.P._LL = [-rt.value for rt in res.trace]
    #m.P._B .= parms[1:m.P.p]
    #m.P._S .= parms[(m.P.p+1):end]
    m.P.X = keepx ? m.P.X : nothing
    m.R = keepy ? m.R : nothing
    m
end

function StatsBase.fit!(
    m::PSModel;
    verbose::Bool = false,
    maxiter::Integer = 1000,
    gtol::Float64 = 1e-8,
    start = nothing,
    kwargs...,
)
    if !issubset(keys(kwargs), (:keepx, :keepy, :bootstrap_sample, :bootstrap_rng))
        throw(ArgumentError("unsupported keyword argument in: $(kwargs...)"))
    end
    _fit!(m, verbose = verbose, maxiter = maxiter, gtol = gtol, start = start; kwargs...)
end



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
    fitint = true,
    fitargs...,
) where {M<:PSModel,Y<:Union{Vector{<:Real},BitVector}}

    # Check that X and y have the same number of observations
    if size(X, 1) != size(y, 1)
        throw(DimensionMismatch("number of rows in X and y must match"))
    end

    R = LSurvivalResp(enter, exit, y, wts, id)
    survcheck(R)
    P0 = PSParms(ones(size(X, 1), 1), extraparms = length(dist) - 1)
    res0 = M(R, P0, dist)
    start0 = LSurvival.setinits(res0)
    fitint && fit!(res0, start = start0)
    #

    P = PSParms(X, extraparms = length(dist) - 1)
    #if !haskey(fitargs, :start)
    #    st = zeros(length(P._grad))
    #    st[1] = params(res0)[1]
    #    st[end] = params(res0)[end]
    #    fitargs = (start = st, fitargs...)
    #end
    res = M(R, P, dist)
    fit!(res; fitargs...)
    fitint && pushfirst!(res.P._LL, res0.P._LL[end])
    return res
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
    id::Union{AbstractVector{<:AbstractLSurvivalID}, Nothing} = nothing,
    wts::Union{AbstractVector,Nothing} = nothing,
    offset::Union{AbstractVector,Nothing}= nothing,
    contrasts::AbstractDict{Symbol} = Dict{Symbol,Any}(),
    fitint = true,
    fitargs...,
) where {M<:PSModel}
    f, (y, X) = modelframe(f, data, contrasts, M)
    
    id = id === nothing ? [ ID(i) for i in eachindex(y) ] : id
    offset = offset === nothing ? similar(getindex(X,[1]), 0) : offset
    wts = wts === nothing ? similar(getindex(X,[1]), 0) : wts


    R = LSurvivalResp(y, wts, id)
    survcheck(R)

    P0 = PSParms(ones(size(X, 1), 1), extraparms = length(dist) - 1)
    res0 = M(R, P0, dist)
    start0 = LSurvival.setinits(res0)
    fitint && fit!(res0, start = start0)

    P = PSParms(X, extraparms = length(dist) - 1)
    #if !haskey(fitargs, :start)
    #    st = zeros(length(P._grad))
    #    st[1] = params(res0)[1]
    #    st[end] = params(res0)[end]
    #    fitargs = (start = st, fitargs...)
    #end
    res = M(R, P, f, dist, false)

    fit!(res; fitargs...)
    fitint && pushfirst!(res.P._LL, res0.P._LL[end])
    return res
end

survreg(X, enter, exit, y, args...; kwargs...) =
    fit(PSModel, X, enter, exit, y, args...; kwargs...)

survreg(f::FormulaTerm, data; kwargs...) = fit(PSModel, f, data; kwargs...)

##################################################################################################################### 
# summary functions for PSModel objects
#####################################################################################################################

params(m::M) where {M<:PSModel} = vcat(m.P._B, m.P._S)
formula(x::M) where {M<:PSModel} = x.formula
logscale(x::M) where {M<:PSModel} = x.P._S
scale(x::M) where {M<:PSModel} = exp.(logscale(x))

StatsBase.coefnames(x::M) where {M<:PSModel} =
    x.formula === nothing ? ["β$i" for i = 1:length(coef(x))] : coefnames(formula(x).rhs)

function StatsBase.coef(m::M) where {M<:PSModel}
    mwarn(m)
    m.P._B
end

# quantile function for standard normal

function StatsBase.coeftable(m::M; level::Float64 = 0.95) where {M<:PSModel}
    mwarn(m)
    #β = vcat(coef(m), scale(m))
    β = params(m)
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
    parmnames = ["$f" for f in fieldnames(typeof(m.d))]
    parmnames = replace.(parmnames, "ρ" => "log(scale)")

    rown = vcat(coefnames(m), [parmnames[i+1] for i = 1:length(scale(m))])
    rown = typeof(rown) <: AbstractVector ? rown : [rown]
    StatsBase.CoefTable(op, head, rown, pcol, zcol)
end



function StatsBase.confint(m::M; level::Float64 = 0.95, kwargs...) where {M<:PSModel}
    mwarn(m)
    beta = params(m)
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

function StatsBase.predict(m::M) where {M<:PSModel}
    exp.(fitted(m))
end

function StatsBase.predict!(m::M) where {M<:PSModel}
    m.P._r .= exp.(fitted(m))
    nothing
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

function StatsBase.vcov(m::M; type::Union{String,Nothing} = nothing, seed=nothing, iter=200) where {M<:PSModel}
    mwarn(m)
    if type == "robust"
        #res = robust_vcov(m)
        throw("$type not implemented")
    elseif type == "jackknife"
        res = jackknife_vcov(m)
        #throw("$type not implemented")
    elseif type == "bootstrap"
        res = bootstrap_vcov(m, iter, seed=seed)
    else
        res = -inv(m.P._hess)
        if any(eigen(res).values .< 0.0)
            @warn(
                "Covariance matrix is not positive semi-definite, model likely not converged"
            )
            if any(diag(res) .< 0.0)
                res = zeros(size(m.P._hess))
            end
        end
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
    # Coefficients
    coeftab = coeftable(m, level = level)
    iob = IOBuffer()
    println(iob, coeftab)
    # Model fit
    ll = loglikelihood(m)
    llnull = nullloglikelihood(m)
    str = """\nMaximum likelihood estimates (alpha=$(@sprintf("%.2g", 1-level))):\n"""
    str *= String(take!(iob))
    str *= "$(nameof(typeof(m.d))) distribution\n"
    str *= "Log-likelihood (full): $(@sprintf("%8g", ll))\n"
    if ll > llnull
        df = length(coeftab.rownms) - length(m.d)
        chi2 = 2 * (ll - llnull)
        lrtp = 1 - cdfchisq(df, chi2)
        str *= "Log-likelihood (Intercept only): $(@sprintf("%8g", llnull))\n"
        str *= "LRT p-value (χ²=$(round(chi2, digits=2)), df=$df): $(@sprintf("%.5g", lrtp))\n"
    end
    str *= "Solver iterations: $(length(m.P._LL)-1)"
    println(io, str)
end

Base.show(m::M; kwargs...) where {M<:PSModel} = Base.show(stdout, m; kwargs...)


"""
using LSurvival, Random, StatsBase, Printf, Tables
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
enter = zeros(length(dat1.time))
t = dat1.time
d = dat1.status
X = hcat(ones(length(dat1.x)), dat1.x)
wt = ones(length(t))

dist = Weibull()
P = PSParms(X[:,1:1], extraparms=length(dist)-1)
P = PSParms(X, extraparms=length(dist)-1)
R = LSurvivalResp(dat1.time, dat1.status)    # specification with ID only
m = PSModel(R,P,dist)

LSurvival._fit!(m);
dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])
survreg(@formula(Surv(time,status)~x), dat1, dist=LSurvival.Weibull())
""";
