##################################################################################################################### 
# structs
#####################################################################################################################



mutable struct PHParms{
    D<:Matrix{<:Real},
    B<:Vector{<:Float64},
    R<:Vector{<:Float64},
    L<:Vector{<:Float64},
    H<:Matrix{<:Float64},
    I<:Int64,
} <: AbstractLSurvParms
    X::Union{Nothing,D}
    _B::B                        # coefficient vector
    _r::R                        # linear predictor/risk
    _LL::L                 # partial likelihood history
    _grad::B     # gradient vector
    _hess::H     # Hessian matrix
    n::I                     # number of observations
    p::I                     # number of parameters
end

function PHParms(
    X::Union{Nothing,D},
    _B::B,
    _r::R,
    _LL::L,
    _grad::B,
    _hess::H,
) where {
    D<:Matrix{<:Real},
    B<:Vector{<:Float64},
    R<:Vector{<:Float64},
    L<:Vector{<:Float64},
    H<:Matrix{<:Float64},
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
    Base.println(io, "Slots: X, _B, _grad, _hess, _r, _n, p\n")
    Base.println(io, "Predictor matrix (X):")
    Base.show(io, "text/plain", x.X)
end
Base.show(x::PHParms) = Base.show(stdout, x::PHParms)


"""
$DOC_PHMODEL    
"""
mutable struct PHModel{G<:LSurvResp,L<:AbstractLSurvParms} <: AbstractPH
    R::Union{Nothing,G}        # Survival response
    P::L        # parameters
    ties::String
    fit::Bool
    bh::AbstractMatrix
end

"""
$DOC_PHMODEL 
"""
function PHModel(
    R::Union{Nothing,G},
    P::L,
    ties::String,
    fit::Bool,
) where {G<:LSurvResp,L<:AbstractLSurvParms}
    return PHModel(R, P, ties, fit, zeros(Float64, length(R.eventtimes), 4))
end

"""
$DOC_PHMODEL  
"""
function PHModel(
    R::Union{Nothing,G},
    P::L,
    ties::String,
) where {G<:LSurvResp,L<:AbstractLSurvParms}
    return PHModel(R, P, ties, false)
end

"""
$DOC_PHMODEL    
"""
function PHModel(R::Union{Nothing,G}, P::L) where {G<:LSurvResp,L<:AbstractLSurvParms}
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
    ntimes::Int64 = size(bh, 1)
    risk, surv = zeros(Float64, ntimes, length(eventtypes)), fill(1.0, ntimes)
    times = bh[:, 4]
    event = bh[:, 5]
    PHSurv(fitlist, eventtypes, times, surv, risk, bh[:, 1], event, false)
end

"""
$DOC_PHSURV
"""
function PHSurv(fitlist::Array{T}) where {T<:PHModel}
    eventtypes = collect(eachindex(fitlist))
    PHSurv(fitlist, eventtypes)
end

##################################################################################################################### 
# fitting functions for PHModel objects
#####################################################################################################################

function _fit!(
    m::PHModel;
    verbose::Bool = false,
    maxiter::Integer = 500,
    atol::Float64 = sqrt(1e-8),
    rtol::Float64 = 1e-8,
    start = nothing,
    keepx = false,
    keepy = false,
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
    lowermethod3 = lowercase(m.ties[1:3])
    # Newton Raphson step size scaler
    λ = 1.0
    #
    totiter = 0
    oldQ = floatmax()
    lastLL = -floatmax()
    risksetidxs, caseidxs = [], []
    @inbounds for _outj in m.R.eventtimes
        push!(risksetidxs, findall((m.R.enter .< _outj) .&& (m.R.exit .>= _outj)))
        #push!(risksetidxs, findall((m.R.enter .< _outj) .&& (m.R.exit .>= _outj))) # implement with strata argument
        push!(
            caseidxs,
            findall((m.R.y .> 0) .&& isapprox.(m.R.exit, _outj) .&& (m.R.enter .< _outj)),
        )
    end
    den, _sumwtriskset, _sumwtcase = _stepcox!(lowermethod3, m, risksetidxs, caseidxs)
    _llhistory = [m.P._LL[1]] # if inits are zero, 2*(_llhistory[end] - _llhistory[1]) is the likelihood ratio test on all predictors
    # repeat newton raphson steps until convergence or max iterations
    while totiter < maxiter
        totiter += 1
        ######
        # update 
        #######
        Q = 0.5 * (m.P._grad' * m.P._grad) #modified step size if gradient increases
        likrat = (lastLL / m.P._LL[1])
        absdiff = abs(lastLL - m.P._LL[1])
        reldiff = max(likrat, inv(likrat)) - 1.0
        converged = (reldiff < atol) || (absdiff < rtol)
        if converged
            break
        end
        if Q > oldQ # gradient has increased, indicating the maximum of a monotonic partial likelihood was overshot
            λ *= 0.5  # step-halving
        else
            λ = min(2.0λ, 1.0) # de-halving
        end
        isnan(m.P._LL[1]) ? throw("Log-partial-likelihood is NaN") : true
        if abs(m.P._LL[1]) != Inf
            m.P._B .+= inv(-(m.P._hess)) * m.P._grad .* λ # newton raphson step
            oldQ = Q
        else
            throw("Log-partial-likelihood is infinite")
        end
        lastLL = m.P._LL[1]
        den, _, _ = _stepcox!(lowermethod3, m, risksetidxs, caseidxs)
        push!(_llhistory, m.P._LL[1])
        verbose ? println(m.P._LL[1]) : true
    end
    if (totiter == maxiter) && (maxiter > 0)
        @warn "Algorithm did not converge after $totiter iterations"
    end
    if verbose && (maxiter == 0)
        @warn "maxiter = 0, model coefficients set to starting values"
    end
    if lowermethod3 == "bre"
        m.bh = [_sumwtcase ./ den _sumwtriskset _sumwtcase m.R.eventtimes]
    elseif lowermethod3 == "efr"
        m.bh = [1.0 ./ den _sumwtriskset _sumwtcase m.R.eventtimes]
    end
    m.P._LL = _llhistory
    m.fit = true
    m.P.X = keepx ? m.P.X : nothing
    m.R = keepy ? m.R : nothing
    m
end

function StatsBase.fit!(
    m::AbstractPH;
    verbose::Bool = false,
    maxiter::Integer = 500,
    atol::Float64 = 1e-6,
    rtol::Float64 = 1e-6,
    start = nothing,
    kwargs...,
)
    if haskey(kwargs, :maxIter)
        Base.depwarn("'maxIter' argument is deprecated, use 'maxiter' instead", :fit!)
        maxiter = kwargs[:maxIter]
    end
    if haskey(kwargs, :convTol)
        Base.depwarn(
            "'convTol' argument is deprecated, use `atol` and `rtol` instead",
            :fit!,
        )
        rtol = kwargs[:convTol]
    end
    if !issubset(keys(kwargs), (:maxIter, :convTol, :tol, :keepx, :keepy))
        throw(ArgumentError("unsupported keyword argument"))
    end
    if haskey(kwargs, :tol)
        Base.depwarn("`tol` argument is deprecated, use `atol` and `rtol` instead", :fit!)
        rtol = kwargs[:tol]
        atol = sqrt(kwargs[:tol])
    end

    start = isnothing(start) ? zeros(Float64, m.P.p) : start

    _fit!(
        m,
        verbose = verbose,
        maxiter = maxiter,
        atol = atol,
        rtol = rtol,
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
    ties = "breslow",
    id::Vector{<:AbstractLSurvID} = [ID(i) for i in eachindex(y)],
    wts::Vector{<:Real} = similar(enter, 0),
    offset::Vector{<:Real} = similar(enter, 0),
    fitargs...,
) where {M<:AbstractPH,Y<:Union{Vector{<:Real},BitVector}}

    # Check that X and y have the same number of observations
    if size(X, 1) != size(y, 1)
        throw(DimensionMismatch("number of rows in X and y must match"))
    end

    R = LSurvResp(enter, exit, y, wts, id)
    P = PHParms(X)

    res = M(R, P, ties)

    return fit!(res; fitargs...)
end


"""
$DOC_FIT_ABSTRACPH
"""
coxph(X, enter, exit, y, args...; kwargs...) =
    fit(PHModel, X, enter, exit, y, args...; kwargs...)


##################################################################################################################### 
# summary functions for PHModel objects
#####################################################################################################################

function StatsBase.coef(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.P._B
end

function StatsBase.coeftable(m::M; level::Float64 = 0.95) where {M<:AbstractPH}
    mwarn(m)
    beta = coef(m)
    std_err = stderror(m)
    z = beta ./ std_err
    zcrit = quantile.(Distributions.Normal(), [(1 - level) / 2, 1 - (1 - level) / 2])
    lci = beta .+ zcrit[1] * std_err
    uci = beta .+ zcrit[2] * std_err
    pval = calcp.(z)
    op = hcat(beta, std_err, lci, uci, z, pval)
    head = ["ln(HR)", "StdErr", "LCI", "UCI", "Z", "P(>|Z|)"]
    rown = ["b$i" for i = 1:size(op)[1]]
    StatsBase.CoefTable(op, head, rown, 6, 5)
end

function StatsBase.confint(m::M; level::Float64 = 0.95) where {M<:AbstractPH}
    mwarn(m)
    beta = coef(m)
    std_err = stderror(m)
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

function StatsBase.loglikelihood(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.P._LL[end]
end

function StatsBase.modelmatrix(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.P.X
end

function StatsBase.nullloglikelihood(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.P._LL[1]
end

function StatsBase.response(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.R
end

function StatsBase.score(m::M) where {M<:AbstractPH}
    mwarn(m)
    m.P._grad
end

function StatsBase.stderror(m::M) where {M<:AbstractPH}
    mwarn(m)
    sqrt.(diag(vcov(m)))
end

function StatsBase.vcov(m::M) where {M<:AbstractPH}
    mwarn(m)
    -inv(m.P._hess)
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
    chi2 = ll - llnull
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
    Base.show(stdout, m::M; kwargs...) where {M<:AbstractPH}

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
    nothing
end


##################################################################################################################### 
# partial likelihood/gradient/hessian functions for tied events
####################################################################################################################

"""
$DOC_LGH_BRESLOW
"""
function lgh_breslow!(_den, m::M, caseidx, risksetidx, j) where {M<:AbstractPH}
    Xcases = m.P.X[caseidx, :]
    Xriskset = m.P.X[risksetidx, :]
    _rcases = m.P._r[caseidx]
    _rriskset = m.P._r[risksetidx]
    _wtcases = m.R.wts[caseidx]
    _wtriskset = m.R.wts[risksetidx]

    den = sum(_rriskset .* _wtriskset)
    m.P._LL .+= sum(_wtcases .* log.(_rcases)) .- log(den) * sum(_wtcases)
    #
    numg = Xriskset' * (_rriskset .* _wtriskset)
    xbar = numg / den # risk-score-weighted average of X columns among risk set
    m.P._grad .+= (Xcases .- xbar')' * (_wtcases)
    #
    numgg = (Xriskset' * Diagonal(_rriskset .* _wtriskset) * Xriskset)
    xxbar = numgg / den
    m.P._hess .+= -(xxbar - xbar * xbar') * sum(_wtcases)
    _den[j] = den
    nothing
end

function efron_weights(m)
    [(l - 1) / m for l = 1:m]
end


"""
$DOC_LGH_EFRON
"""
function lgh_efron!(_den, m::M, caseidx, risksetidx, j, nties) where {M<:AbstractPH}
    Xcases = m.P.X[caseidx, :]
    Xriskset = m.P.X[risksetidx, :]
    _rcases = m.P._r[caseidx]
    _rriskset = m.P._r[risksetidx]
    _wtcases = m.R.wts[caseidx]
    _wtriskset = m.R.wts[risksetidx]

    effwts = efron_weights(nties)
    den = sum(_wtriskset .* _rriskset)
    denc = sum(_wtcases .* _rcases)
    dens = [den - denc * ew for ew in effwts]
    m.P._LL .+= sum(_wtcases .* log.(_rcases)) .- sum(log.(dens)) * 1 / nties * sum(_wtcases) # gives same answer as R with weights
    #
    numg = Xriskset' * (_wtriskset .* _rriskset)
    numgs = [numg .- ew * Xcases' * (_wtcases .* _rcases) for ew in effwts]
    xbars = numgs ./ dens # risk-score-weighted average of X columns among risk set
    m.P._grad .+= Xcases' * _wtcases
    for i = 1:nties
        m.P._grad .+= (-xbars[i]) * sum(_wtcases) / nties
    end
    numgg = (Xriskset' * Diagonal(_wtriskset .* _rriskset) * Xriskset)
    numggs =
        [numgg .- ew .* Xcases' * Diagonal(_wtcases .* _rcases) * Xcases for ew in effwts]
    xxbars = numggs ./ dens
    #
    for i = 1:nties
        m.P._hess .-= (xxbars[i] - xbars[i] * xbars[i]') .* sum(_wtcases) / nties
    end
    #_den[j] = den # Breslow estimator
    sw = sum(_wtcases)
    aw = sw / nties
    _den[j] = 1.0 ./ sum(aw ./ dens) # using Efron estimator
    nothing
end

"""
$DOC_LGH
"""
#function lgh!(lowermethod3, _den, _LL, _grad, _hess, j, p, X, _r, _wt, caseidx, risksetidx)
function lgh!(lowermethod3, _den, m::M, j, caseidx, risksetidx) where {M<:AbstractPH}
    if lowermethod3 == "efr"
        lgh_efron!(_den, m, caseidx, risksetidx, j, length(caseidx))
    elseif lowermethod3 == "bre"
        lgh_breslow!(_den, m, caseidx, risksetidx, j)
    end
end

"""
$DOC__STEPCOXi
"""
function _stepcox!(
    lowermethod3::String,
    m::M,
    # big indexes
    risksetidxs,
    caseidxs,
) where {M<:AbstractPH}
    #_coxrisk!(_r, X, _B) # updates all elements of _r as exp(X*_B)
    _coxrisk!(m.P) # updates all elements of _r as exp(X*_B)
    # loop over event times
    ne = length(m.R.eventtimes)
    den, wtdriskset, wtdcases = zeros(ne), zeros(ne), zeros(ne)
    m.P._LL .*= 0.0
    m.P._grad .*= 0.0
    m.P._hess .*= 0.0
    @inbounds for j = 1:ne
        risksetidx = risksetidxs[j]
        caseidx = caseidxs[j]
        lgh!(lowermethod3, den, m, j, caseidx, risksetidx)
        wtdriskset[j] = sum(m.R.wts[risksetidx])
        wtdcases[j] = sum(m.R.wts[caseidx])
    end # j
    den, wtdriskset, wtdcases
end #function _stepcox!

##################################################################################################################### 
# fitting functions for PHSurv objects
#####################################################################################################################

function _fit!(m::M; coef_vectors = nothing, pred_profile = nothing) where {M<:PHSurv}
    hr = ones(Float64, length(m.eventtypes))
    ch::Float64 = 0.0
    lsurv::Float64 = 1.0
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
