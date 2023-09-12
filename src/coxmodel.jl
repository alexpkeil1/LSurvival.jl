##################################################################################################################### 
# structs
#####################################################################################################################

mutable struct PHParms{
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
    _LL::L                 # partial likelihood history
    _grad::G     # gradient vector
    _hess::H     # Hessian matrix
    n::I                     # number of observations
    p::I                     # number of parameters
end

function PHParms(
    X::Union{Nothing,D},
    _B::B,
    _r::R,
    _LL::L,
    _grad::G,
    _hess::H,
) where {
    D<:Matrix{<:Real},
    B<:Vector{<:Float64},
    R<:Vector{<:Float64},
    L<:Vector{<:Float64},
    G<:Union{Nothing,Vector{<:Float64}},
    H<:Union{Nothing,Matrix{<:Float64}},
}
    n = length(_r)
    p = length(_B)
    return PHParms(X, _B, _r, _LL, _grad, _hess, n, p)
end

function PHParms(X::Union{Nothing,D}) where {D<:AbstractMatrix}
    n, p = size(X)
    PHParms(X, fill(0.0, p), fill(0.0, n), zeros(Float64, 1), fill(0.0, p), fill(0.0, p, p))
end

function Base.show(io::IO, x::PHParms)
    Base.println(io, "Slots: X, _B, _grad, _hess, _r, n, p\n")
    Base.println(io, "Predictor matrix (X):")
    Base.show(io, "text/plain", x.X)
end
Base.show(x::PHParms) = Base.show(stdout, x::PHParms)

Base.size(x::PHParms) = size(x.X)

"""
$DOC_PHMODEL    
"""
mutable struct PHModel{G<:LSurvivalResp,L<:AbstractLSurvivalParms} <: AbstractPH
    R::Union{Nothing,G}        # Survival response
    P::L        # parameters
    formula::Union{FormulaTerm,Nothing}
    ties::String
    fit::Bool
    bh::Matrix{Float64}
    RL::Union{Nothing,Vector{Matrix{Float64}}}        # residual matrix    
end

"""
$DOC_PHMODEL 
"""
function PHModel(
    R::Union{Nothing,G},
    P::L,
    formula::Union{FormulaTerm,Nothing},
    ties::String,
    fit::Bool,
    bh::Matrix{Float64},
) where {G<:LSurvivalResp,L<:AbstractLSurvivalParms}
    tl = ["efron", "breslow"]
    if !issubset([ties], tl)
        jtl = join(tl, ", ")
        throw("`ties` must be one of: $jtl")
    end
    return PHModel(R, P, formula, ties, fit, bh, nothing)
end

"""
$DOC_PHMODEL 
"""
function PHModel(
    R::Union{Nothing,G},
    P::L,
    formula::Union{FormulaTerm,Nothing},
    ties::String,
    fit::Bool,
) where {G<:LSurvivalResp,L<:AbstractLSurvivalParms}
    return PHModel(R, P, formula, ties, fit, zeros(Float64, length(R.eventtimes), 6))
end

"""
$DOC_PHMODEL 
"""
function PHModel(
    R::Union{Nothing,G},
    P::L,
    formula::Union{FormulaTerm,Nothing},
    ties::String,
) where {G<:LSurvivalResp,L<:AbstractLSurvivalParms}
    return PHModel(R, P, formula, ties, false)
end

"""
$DOC_PHMODEL  
"""
function PHModel(
    R::Union{Nothing,G},
    P::L,
    ties::String,
) where {G<:LSurvivalResp,L<:AbstractLSurvivalParms}
    return PHModel(R, P, nothing, ties, false)
end

"""
$DOC_PHMODEL    
"""
function PHModel(
    R::Union{Nothing,G},
    P::L,
    formula::Union{FormulaTerm,Nothing},
) where {G<:LSurvivalResp,L<:AbstractLSurvivalParms}
    return PHModel(R, P, formula, "efron", false)
end

"""
$DOC_PHMODEL    
"""
function PHModel(R::Union{Nothing,G}, P::L) where {G<:LSurvivalResp,L<:AbstractLSurvivalParms}
    return PHModel(R, P, "efron")
end

"""
$DOC_PHSURV
"""
mutable struct PHSurv{G<:Array{T} where {T<:PHModel}} <: AbstractNPSurv
    fitlist::G        # Survival response
    eventtypes::Vector
    times::Vector
    surv::Vector{Float64}
    risk::Matrix{Float64}
    basehaz::Vector{Float64}
    event::Vector{Float64}
    fit::Bool
end

"""
$DOC_PHSURV
"""
function PHSurv(fitlist::Array{T}, eventtypes) where {T<:PHModel}
    bhlist = [ft.bh for ft in fitlist]
    bhlist = [hcat(bh, fill(eventtypes[i], size(bh, 1))) for (i, bh) in enumerate(bhlist)]
    bh = reduce(vcat, bhlist)
    sp = sortperm(bh[:, 4])
    bh = bh[sp, :]
    ntimes::Int = size(bh, 1)
    risk, surv = zeros(Float64, ntimes, length(eventtypes)), fill(1.0, ntimes)
    times = bh[:, 4]
    event = bh[:, end]
    PHSurv(fitlist, eventtypes, times, surv, risk, bh[:, 1], event, false)
end

"""
$DOC_PHSURV
"""
function PHSurv(fitlist::Array{T}) where {T<:PHModel}
    eventtypes = collect(eachindex(fitlist))
    PHSurv(fitlist, eventtypes)
end

StatsModels.drop_intercept(::Type{T}) where {T<:PHModel} = true

##################################################################################################################### 
# fitting functions for PHModel objects
#####################################################################################################################

function _fit!(
    m::PHModel;
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
    start = isnothing(start) ? zeros(length(m.P._B)) : start
    m.P._B = start
    if haskey(kwargs, :ties)
        m.ties = kwargs[:ties]
    end
    # Newton Raphson step size scaler
    λ = 1.0
    #
    totiter = 0
    oldQ = floatmax()
    lastLL = -floatmax()
    ne = length(m.R.eventtimes)
    risksetidxs, caseidxs =
        Array{Array{Int,1},1}(undef, ne), Array{Array{Int,1},1}(undef, ne)
    #den, _sumwtriskset, _sumwtcase =
    #    zeros(Float64, ne), zeros(Float64, ne), zeros(Float64, ne)
    #@inbounds for j = 1:ne
    @inbounds @simd for j = 1:ne
        _outj = m.R.eventtimes[j]
        fr = findall((m.R.enter .< _outj) .&& (m.R.exit .>= _outj))
        fc = findall((m.R.y .> 0) .&& isapprox.(m.R.exit, _outj) .&& (m.R.enter .< _outj))
        risksetidxs[j] = fr
        caseidxs[j] = fc
    end
    # cox risk and set to zero were both in step cox - return them?
    # loop over event times
    _update_PHParms!(m, ne, caseidxs, risksetidxs)
    _llhistory = [m.P._LL[1]] # if inits are zero, 2*(_llhistory[end] - _llhistory[1]) is the likelihood ratio test on all predictors
    # repeat newton raphson steps until convergence or max iterations
    while totiter < maxiter
        totiter += 1
        # check convergence using infinite norm of gradient
        converged = (maximum(abs.(m.P._grad)) < gtol)
        if converged
            break
        end
        # check for descent and modify step size if needed
        #Q = 0.5 * (m.P._grad' * m.P._grad) #modified step size if gradient increases
        Q = m.P._grad' * m.P._grad #l2 norm of vector
        if Q > oldQ # gradient has increased, indicating the maximum  was overshot
            λ *= 0.5  # step-halving
        else
            λ = min(2.0λ, 1.0) # de-halving
        end
        isnan(m.P._LL[1]) ?
        throw("Log-partial-likelihood is NaN: try different starting values") : true
        if abs(m.P._LL[1]) != Inf
            m.P._B .+= inv(-(m.P._hess)) * m.P._grad .* λ # newton raphson step
            oldQ = Q
        else
            @debug "Log-partial-likelihood history: $_llhistory $(m.P._LL[1])"
            throw("Log-partial-likelihood is not finite: check model inputs")
        end
        lastLL = m.P._LL[1]
        _update_PHParms!(m, ne, caseidxs, risksetidxs)
        push!(_llhistory, m.P._LL[1])
        verbose ? println(m.P._LL[1]) : true
    end
    if (totiter == maxiter) && (maxiter > 0)
        @warn "Algorithm did not converge after $totiter iterations: check for collinearity of predictors"
        @debug "recent log-partial-likelihood history: $(_llhistory[end-max(10,maxiter-1):end]) $(m.P._LL[1])"
    end
    if verbose && (maxiter == 0)
        @warn "maxiter = 0, model coefficients set to starting values"
    end
    m.P._LL = _llhistory
    m.fit = true
    basehaz!(m)
    m.P.X = keepx ? m.P.X : nothing
    m.R = keepy ? m.R : nothing
    m
end

function StatsBase.fit!(
    m::AbstractPH;
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
    if !issubset(keys(kwargs), (:maxIter, :convTol, :tol, :keepx, :keepy))
        throw(ArgumentError("unsupported keyword argument"))
    end
    if haskey(kwargs, :tol)
        Base.depwarn("`tol` argument is deprecated, use `gtol` instead", :fit!)
        gtol = kwargs[:tol]
    end

    start = isnothing(start) ? zeros(Float64, m.P.p) : start

    _fit!(
        m,
        verbose = verbose,
        maxiter = maxiter,
        gtol = gtol,
        start = start;
        kwargs...,
    )
end


"""
$DOC_FIT_ABSTRACPH
"""
function fit(
    ::Type{M},
    X::Matrix{<:Real},#{<:FP},
    enter::Vector{<:Real},
    exit::Vector{<:Real},
    y::Y;
    ties = "efron",
    id::Vector{<:AbstractLSurvivalID} = [ID(i) for i in eachindex(y)],
    wts::Vector{<:Real} = similar(enter, 0),
    offset::Vector{<:Real} = similar(enter, 0),
    fitargs...,
) where {M<:AbstractPH,Y<:Union{Vector{<:Real},BitVector}}

    # Check that X and y have the same number of observations
    if size(X, 1) != size(y, 1)
        throw(DimensionMismatch("number of rows in X and y must match"))
    end

    R = LSurvivalResp(enter, exit, y, wts, id)
    P = PHParms(X)

    res = M(R, P, ties)

    return fit!(res; fitargs...)
end


function modelframe(
    f::FormulaTerm,
    data,
    contrasts::AbstractDict,
    ::Type{M},
) where {M<:AbstractPH}
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
    ties = "efron",
    id::AbstractVector{<:AbstractLSurvivalID} = [ID(i) for i in eachindex(getindex(data, 1))],
    wts::AbstractVector{<:Real} = similar(getindex(data, 1), 0),
    offset::AbstractVector{<:Real} = similar(getindex(data, 1), 0),
    contrasts::AbstractDict{Symbol} = Dict{Symbol,Any}(),
    fitargs...,
) where {M<:AbstractPH}
    f, (y, X) = modelframe(f, data, contrasts, M)
    R = LSurvivalResp(y, wts, id)
    P = PHParms(X)
    res = M(R, P, f, ties)
    return fit!(res; fitargs...)
end


"""
$DOC_FIT_ABSTRACPH
"""
coxph(X, enter, exit, y, args...; kwargs...) =
    fit(PHModel, X, enter, exit, y, args...; kwargs...)

coxph(f::FormulaTerm, data; kwargs...) = fit(PHModel, f, data; kwargs...)


##################################################################################################################### 
# summary functions for PHModel objects
#####################################################################################################################

formula(x::M) where {M<:AbstractPH} = x.formula

StatsBase.coefnames(x::M) where {M<:AbstractPH} =
    x.formula === nothing ? ["b$i" for i = 1:length(coef(x))] : coefnames(formula(x).rhs)

function StatsBase.coef(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.P._B
end

function StatsBase.coeftable(m::M; level::Float64 = 0.95) where {M<:AbstractPH}
    mwarn(m)
    beta = coef(m)
    std_err = stderror(m)
    zcrit = quantile.(Distributions.Normal(), [(1 - level) / 2, 1 - (1 - level) / 2])
    lci = beta .+ zcrit[1] * std_err
    uci = beta .+ zcrit[2] * std_err
    z = beta ./ std_err
    pval = calcp.(z)
    op = hcat(beta, std_err, lci, uci, z, pval)
    head = ["ln(HR)", "StdErr", "LCI", "UCI", "Z", "P(>|Z|)"]
    pcol=6
    zcol=5
    if !isnothing(m.RL)
        std_err_rob = stderror(m, type = "robust")
        z = beta ./ std_err_rob
        pval = calcp.(z)
        lci = beta .+ zcrit[1] * std_err_rob
        uci = beta .+ zcrit[2] * std_err_rob
        op = hcat(beta, std_err, std_err_rob, lci, uci, z, pval)
        pcol += 1
        zcol += 1
        head = ["ln(HR)", "StdErr", "RobustSE", "LCI", "UCI", "Z", "P(>|Z|)"]
    end
    #rown = ["b$i" for i = 1:size(op)[1]]
    rown = coefnames(m)
    rown = length(rown) > 1 ? rown : [rown]
    StatsBase.CoefTable(op, head, rown, pcol, zcol)
end



"""
$DOC_CONFINT
"""
function StatsBase.confint(m::M; level::Float64 = 0.95, kwargs...) where {M<:AbstractPH}
    mwarn(m)
    beta = coef(m)
    std_err = stderror(m; kwargs...) # can have type="robust"
    z = beta ./ std_err
    zcrit = quantile.(Distributions.Normal(), [(1 - level) / 2, 1 - (1 - level) / 2])
    lci = beta .+ zcrit[1] * std_err
    uci = beta .+ zcrit[2] * std_err
    hcat(lci, uci)
end

function StatsBase.fitted(m::M) where {M<:AbstractPH}
    mwarn(m)
    D = modelmatrix(m)
    D * coef(m)
end

function StatsBase.isfitted(m::M) where {M<:AbstractPH}
    m.fit
end

function StatsBase.deviance(m::M) where {M<:AbstractPH}
    mwarn(m)
    -2 * loglikelihood(m)
end

function StatsBase.aic(m::M) where {M<:AbstractPH}
    deviance(m) + 2 * dof(m)
end

function StatsBase.aicc(m::M) where {M<:AbstractPH}
    df = dof(m)
    aic(m) + 2 * df * (df - 1) / (dof_residual(m) - 1)
end

function StatsBase.bic(m::M) where {M<:AbstractPH}
    deviance(m) + dof(m) * log(nobs(m))
end


function StatsBase.nulldeviance(m::M) where {M<:AbstractPH}
    mwarn(m)
    -2 * nullloglikelihood(m)
end

function StatsBase.dof(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.P.p
end

function StatsBase.dof_residual(m::M) where {M<:AbstractPH}
    mwarn(m)
    nobs(m) - dof(m)
end


function StatsBase.nobs(m::M) where {M<:AbstractPH}
    mwarn(m)
    length(unique(m.R.id))
end

"""
Maximum log partial likelihood for a fitted `AbstractPH` model
Efron or Breslow (depending on the `ties`` parameter)
"""
function StatsBase.loglikelihood(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.P._LL[end]
end
logpartiallikelihood(m::M) where {M<:AbstractPH} = loglikelihood(m)

function StatsBase.modelmatrix(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.P.X
end

"""
Null log-partial likelihood for a fitted `AbstractPH` model
Efron or Breslow (depending on the `ties`` parameter)

Note: this is just the log partial likelihood at the initial values of the model, which default to 0. If initial values are non-null, then this function no longer validly returns the null log-partial likelihood.
"""
function StatsBase.nullloglikelihood(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.P._LL[1]
end
nulllogpartiallikelihood(m::M) where {M<:AbstractPH} = nullloglikelihood(m)

function StatsBase.model_response(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.R
end

StatsBase.response(m::M) where {M<:AbstractPH} = StatsBase.model_response(m)

function StatsBase.score(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.P._grad
end

function StatsBase.stderror(m::M; kwargs...) where {M<:AbstractPH}
    mwarn(m)
    sqrt.(diag(vcov(m; kwargs...)))
end

"""
$DOC_VCOV
"""
function StatsBase.vcov(m::M; type::Union{String,Nothing} = nothing) where {M<:AbstractPH}
    mwarn(m)
    if type == "robust"
        res = robust_vcov(m)
    else
        res = -inv(m.P._hess)
    end
    res
end

function StatsBase.weights(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.R.wts
end

function Base.show(io::IO, m::M; level::Float64 = 0.95) where {M<:AbstractPH}
    if !m.fit
        println(io, "Model not yet fitted")
        return nothing
    end
    ll = loglikelihood(m)
    llnull = nullloglikelihood(m)
    chi2 = 2 * (ll - llnull)
    coeftab = coeftable(m, level = level)
    df = length(coeftab.rownms)
    lrtp = 1 - cdf(Distributions.Chisq(df), chi2)
    iob = IOBuffer()
    println(iob, coeftab)
    str = """\nMaximum partial likelihood estimates (alpha=$(@sprintf("%.2g", 1-level))):\n"""
    str *= String(take!(iob))
    str *= "Partial log-likelihood (null): $(@sprintf("%8g", llnull))\n"
    str *= "Partial log-likelihood (fitted): $(@sprintf("%8g", ll))\n"
    str *= "LRT p-value (X^2=$(round(chi2, digits=2)), df=$df): $(@sprintf("%.5g", lrtp))\n"
    str *= "Newton-Raphson iterations: $(length(m.P._LL)-1)"
    println(io, str)
end

Base.show(m::M; kwargs...) where {M<:AbstractPH} =
    Base.show(stdout, m; kwargs...)

##################################################################################################################### 
# helper functions
####################################################################################################################

function mwarn(m)
    if !isfitted(m)
        @warn "Model not yet fitted"
    end
end

calcp(z) = (1.0 - cdf(Distributions.Normal(), abs(z))) * 2

function _coxrisk!(p::P) where {P<:PHParms}
    map!(z -> exp(z), p._r, p.X * p._B)
    #p._r .= exp.(p.X * p._B)
    nothing
end


##################################################################################################################### 
# partial likelihood/gradient/hessian functions for tied events
####################################################################################################################

"""
$DOC_LGH_BRESLOW
"""
function lgh_breslow!(m::M, j, caseidx, risksetidx) where {M<:AbstractPH}
    Xcases = view(m.P.X, caseidx, :)
    Xriskset = view(m.P.X, risksetidx, :)
    _rcases = view(m.P._r, caseidx)
    _rriskset = view(m.P._r, risksetidx)
    _wtcases = view(m.R.wts, caseidx)
    _wtriskset = view(m.R.wts, risksetidx)

    rw = _rriskset .* _wtriskset
    sw = sum(_wtcases)
    _den = sum(rw)
    if !isnothing(m.P._LL)
        #m.P._LL .+= sum(_wtcases .* log.(_rcases)) .- log(_den) * sw
        n = length(_wtcases)
        m.P._LL[1] -= log(_den) * sw
        for i = 1:n
            m.P._LL[1] += _wtcases[i] * log(_rcases[i])
        end
    end
    #
    if !isnothing(m.P._grad)
        numg = Xriskset' * rw
        xbar = numg / _den # risk-score-weighted average of X columns among risk set
        m.P._grad .+= (Xcases .- xbar')' * _wtcases
        #
    end
    if !isnothing(m.P._hess)
        numgg = (Xriskset' * Diagonal(rw) * Xriskset)
        xxbar = numgg / _den
        m.P._hess .+= -(xxbar - xbar * xbar') * sw
    end
    nothing
end

function efron_weights(m)
    [(l - 1) / m for l = 1:m]
end

"""
$DOC_LGH_EFRON
"""
function lgh_efron!(m::M, j, caseidx, risksetidx) where {M<:AbstractPH}
    nties = length(caseidx)
    Xcases = view(m.P.X, caseidx, :)
    Xriskset = view(m.P.X, risksetidx, :)
    _rcases = view(m.P._r, caseidx)
    _rriskset = view(m.P._r, risksetidx)
    _wtcases = view(m.R.wts, caseidx)
    _wtriskset = view(m.R.wts, risksetidx)

    sw = sum(_wtcases)
    aw = sw / nties

    effwts = efron_weights(nties)
    WR = _wtriskset .* _rriskset
    WC = _wtcases .* _rcases
    deni = sum(WR)
    denc = sum(WC)
    dens = [deni - denc * ew for ew in effwts]
    if !isnothing(m.P._LL)
        m.P._LL .+=
            sum(_wtcases .* log.(_rcases)) .- sum(log.(dens)) * 1 / nties * sum(_wtcases)
    end
    if !isnothing(m.P._grad)
        numg = Xriskset' * (WR)
        XtW = Xcases' * WC
        numgs = [numg .- ew * XtW for ew in effwts]
        xbars = numgs ./ dens # risk-score-weighted average of X columns among risk set
        m.P._grad .+= Xcases' * _wtcases
        m.P._grad .-= sum(xbars) * aw
    end
    if !isnothing(m.P._hess)
        numgg = Xriskset' * Diagonal(WR) * Xriskset
        XtWX = Xcases' * Diagonal(WC) * Xcases
        numggs = [numgg .- ew .* XtWX for ew in effwts]
        xxbars = numggs ./ dens
        for i = 1:nties
            m.P._hess .-= (xxbars[i] - xbars[i] * xbars[i]') * aw
        end
    end
    nothing
end


"""
$DOC_LGH
"""
#function lgh!(lowermethod3, _den, _LL, _grad, _hess, j, p, X, _r, _wt, caseidx, risksetidx)
function lgh!(m::M, j, caseidx, risksetidx) where {M<:AbstractPH}
    if m.ties == "efron"
        lgh_efron!(m, j, caseidx, risksetidx)
    elseif m.ties == "breslow"
        lgh_breslow!(m, j, caseidx, risksetidx)
    end
end


function _initializeobjective!(P::PHParms)
    fill!(P._LL, 0.0)
    fill!(P._grad, 0.0)
    fill!(P._hess, 0.0)
end

"""
$DOC__UPDATE_PHPARMS
_update_PHParms!(m, risksetidxs, caseidxs, ne, den)
"""
function _update_PHParms!(
    m::M,
    # big indexes
    ne::I,
    caseidxs::Vector{Vector{T}},
    risksetidxs::Vector{Vector{T}},
) where {M<:AbstractPH,I<:Int,T<:Int}
    _coxrisk!(m.P) # updates all elements of _r as exp(X*_B)
    _initializeobjective!(m.P)
    @inbounds @simd for j = 1:ne
        lgh!(m, j, caseidxs[j], risksetidxs[j])
    end # j
    m.P._LL
end #function _update_PHParms!


##################################################################################################################### 
# baseline
#####################################################################################################################

function basehaz!(m::M) where {M<:PHModel}
    ne = length(m.R.eventtimes)
    risksetidxs, caseidxs =
        Array{Array{Int,1},1}(undef, ne), Array{Array{Int,1},1}(undef, ne)
    #    
    den, _sumwtriskset, _sumwtcase, _sumuwtriskset, _sumuwtcase = zeros(Float64, ne),
    zeros(Float64, ne),
    zeros(Float64, ne),
    zeros(Float64, ne),
    zeros(Float64, ne)
    @inbounds @simd for j = 1:ne
        _outj = m.R.eventtimes[j]
        risksetidx = findall((m.R.enter .< _outj) .&& (m.R.exit .>= _outj))
        caseidx =
            findall((m.R.y .> 0) .&& isapprox.(m.R.exit, _outj) .&& (m.R.enter .< _outj))
        nties = length(caseidx)
        denj!(den, m.P._r, m.R.wts, m.ties, caseidx, risksetidx, nties, j)
        _sumwtriskset[j] = sum(m.R.wts[risksetidx])
        _sumwtcase[j] = sum(m.R.wts[caseidx])
        _sumuwtriskset[j] = length(risksetidx)
        _sumuwtcase[j] = length(caseidx)
    end
    if m.ties == "breslow"
        m.bh =
            [_sumwtcase ./ den _sumwtriskset _sumwtcase m.R.eventtimes _sumuwtriskset _sumuwtcase]
    elseif m.ties == "efron"
        m.bh =
            [1.0 ./ den _sumwtriskset _sumwtcase m.R.eventtimes _sumuwtriskset _sumuwtcase]
    end
end

function denj!(den, _r, wts, method, caseidx, risksetidx, nties, j)
    _rcases = view(_r, caseidx)
    _rriskset = view(_r, risksetidx)
    _wtcases = view(wts, caseidx)
    _wtriskset = view(wts, risksetidx)
    deni = sum(_wtriskset .* _rriskset)
    if method == "breslow"
        den[j] = deni
    elseif method == "efron"
        effwts = efron_weights(nties)
        sw = sum(_wtcases)
        aw = sw / nties
        denc = sum(_wtcases .* _rcases)
        dens = [deni - denc * ew for ew in effwts]
        den[j] = 1.0 ./ sum(aw ./ dens) # using Efron estimator
    end
end



##################################################################################################################### 
# fitting functions for PHSurv objects
#####################################################################################################################

function _fit!(m::M; coef_vectors = nothing, pred_profile = nothing) where {M<:PHSurv}
    hr = ones(Float64, length(m.eventtypes))
    ch::Float64 = 0.0
    lsurv::Float64 = 1.0
    if (isnothing(coef_vectors))
        coef_vectors = [coef(fit) for fit in m.fitlist]
    end
    if (!isnothing(coef_vectors) && !isnothing(pred_profile))
        @inbounds for (j, d) in enumerate(m.eventtypes)
            hr[j] = exp(dot(pred_profile, coef_vectors[j]))
        end
    end
    lci = zeros(length(m.eventtypes))
    @inbounds for i in eachindex(m.basehaz)
        @inbounds for (j, d) in enumerate(m.eventtypes)
            if m.event[i] == d
                m.basehaz[i] *= hr[j]                        # baseline hazard times hazard ratio
                m.risk[i, j] = lci[j] + m.basehaz[i] * lsurv
            else
                m.risk[i, j] = lci[j]
            end
        end
        ch += m.basehaz[i]
        m.surv[i] = exp(-ch)
        lsurv = m.surv[i]
        lci = m.risk[i, :]
    end
    m.fit = true
    m
end

"""
$DOC_FIT_PHSURV   
"""
function fit(::Type{M}, fitlist::Vector{<:T}, ; fitargs...) where {M<:PHSurv,T<:PHModel}

    res = M(fitlist)

    return fit!(res; fitargs...)
end

"""
$DOC_FIT_PHSURV
"""
risk_from_coxphmodels(fitlist::Array{T}, args...; kwargs...) where {T<:PHModel} =
    fit(PHSurv, fitlist, args...; kwargs...)


##################################################################################################################### 
# summary functions for PHSurv objects
#####################################################################################################################


function Base.show(io::IO, m::M; maxrows = 20) where {M<:PHSurv}
    if !m.fit
        println("Survival not yet calculated (use fit function)")
        return ""
    end
    types = m.eventtypes
    ev = ["# events (j=$jidx)" for (jidx, j) in enumerate(types)]
    rr = ["risk (j=$jidx)" for (jidx, j) in enumerate(types)]

    resmat = hcat(m.times, m.surv, m.event, m.basehaz, m.risk)
    head = ["time", "survival", "event type", "cause-specific hazard", rr...]
    nr = size(resmat)[1]
    rown = ["$i" for i = 1:nr]

    op = CoefTable(resmat, head, rown)
    iob = IOBuffer()
    if nr < maxrows
        println(iob, op)
    else
        len = floor(Int, maxrows / 2)
        op1, op2 = deepcopy(op), deepcopy(op)
        op1.rownms = op1.rownms[1:len]
        op1.cols = [c[1:len] for c in op1.cols]
        op2.rownms = op2.rownms[(end-len+1):end]
        op2.cols = [c[(end-len+1):end] for c in op2.cols]
        println(iob, op1)
        println(iob, "...")
        println(iob, op2)
    end
    str = """\nCox-model based survival, risk, baseline cause-specific hazard\n"""
    str *= String(take!(iob))
    for (jidx, j) in enumerate(types)
        str *= "Number of events (j=$j): $(@sprintf("%8g", sum(m.event .== m.eventtypes[jidx])))\n"
    end
    str *= "Number of unique event times: $(@sprintf("%8g", length(m.times)))\n"
    println(io, str)
end

Base.show(m::M; kwargs...) where {M<:PHSurv} =
    Base.show(stdout, m::M; kwargs...) where {M<:PHSurv}
