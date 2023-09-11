
##################################################################################################################### 
# structs
#####################################################################################################################

mutable struct KMSurv{G<:LSurvivalResp} <: AbstractNPSurv
    R::Union{Nothing,G}        # Survival response
    times::Vector{<:Real}
    surv::Vector{<:Float64}
    riskset::Vector{<:Real}
    events::Vector{<:Real}
    fit::Bool
end

function KMSurv(R::Union{Nothing,G}) where {G<:LSurvivalResp}
    times = R.eventtimes
    nt = length(times)
    surv = ones(Float64, nt)
    riskset = zeros(Float64, nt)
    events = zeros(Float64, nt)
    KMSurv(R, times, surv, riskset, events, false)
end

mutable struct AJSurv{G<:LSurvivalCompResp} <: AbstractNPSurv
    R::Union{Nothing,G}        # Survival response
    times::Vector{<:Real}
    surv::Vector{<:Float64}
    risk::Matrix{<:Float64}
    riskset::Vector{<:Real}
    events::Matrix{<:Real}
    fit::Bool
end

function AJSurv(R::Union{Nothing,G}) where {G<:LSurvivalCompResp}
    times = R.eventtimes
    net = length(R.eventtypes) - 1
    nt = length(times)
    surv = ones(Float64, nt)
    risk = zeros(Float64, nt, net)
    riskset = zeros(Float64, nt)
    events = zeros(Float64, nt, net)
    AJSurv(R, times, surv, risk, riskset, events, false)
end


##################################################################################################################### 
# Fitting functions for non-parametric survival models
#####################################################################################################################

function _fit!(m::KMSurv; eps = 0.00000001, censval = 0, keepy = true, kwargs...)
    # there is some bad floating point issue with epsilon that should be tracked
    # R handles this gracefully
    # ties allowed
    #_dt = zeros(length(orderedtimes))
    _1mdovern = ones(length(m.times))
    for (_i, tt) in enumerate(m.times)
        R = findall((m.R.exit .>= tt) .& (m.R.enter .< (tt - eps))) # risk set index (if in times are very close to other out-times, not using epsilon will make risk sets too big)
        ni = sum(m.R.wts[R]) # sum of weights in risk set
        di = sum(m.R.wts[R] .* (m.R.y[R] .> censval) .* (m.R.exit[R] .== tt))
        m.events[_i] = di
        _1mdovern[_i] = log(1.0 - di / ni)
        m.riskset[_i] = ni
    end
    m.surv = exp.(cumsum(_1mdovern))
    m.R = keepy ? m.R : nothing
    m.fit = true
    m
end

function _fit!(m::AJSurv; keepy = true, eps = 0.00000001)
    dvalues = m.R.eventtypes[2:end]
    nvals = length(dvalues)
    kmfit = fit(KMSurv, m.R.enter, m.R.exit, m.R.y, wts = m.R.wts)
    m.surv = kmfit.surv
    # overall survival via Kaplan-Meier
    orderedtimes, S, riskset = kmfit.times, kmfit.surv, kmfit.riskset
    Sm1 = vcat(1.0, S)
    for (_i, tt) in enumerate(orderedtimes)
        R = findall((m.R.exit .>= tt) .& (m.R.enter .< (tt - eps))) # risk set
        weightsR = m.R.wts[R]
        ni = sum(weightsR) # sum of weights/weighted individuals in risk set
        m.riskset[_i] = ni
        for (jidx, j) in enumerate(dvalues)
            dij = sum(weightsR .* m.R.eventmatrix[R, jidx] .* (m.R.exit[R] .== tt))
            m.events[_i, jidx] = dij
            m.risk[_i, jidx] = Sm1[_i] * dij / ni
        end
    end
    for jidx = 1:nvals
        m.risk[:, jidx] = cumsum(m.risk[:, jidx])
    end
    m.fit = true
    m.R = keepy ? m.R : nothing
    m
    #orderedtimes, S, ajest, riskset
end;


"""
$DOC_FIT_KMSURV

$DOC_FIT_AJSURV

$DOC_FIT_PHSURV
"""
function StatsBase.fit!(m::T; kwargs...) where {T<:AbstractNPSurv}
    _fit!(m; kwargs...)
end

"""
$DOC_FIT_KMSURV
"""
function fit(
    ::Type{M},
    enter::Vector{<:Real},
    exit::Vector{<:Real},
    y::Y;
    wts::Vector{<:Real} = similar(enter, 0),
    id::Vector{<:AbstractLSurvivalID} = [ID(i) for i in eachindex(y)],
    offset::Vector{<:Real} = similar(enter, 0),
    fitargs...,
) where {M<:KMSurv,Y<:Union{Vector{<:Real},BitVector}}

    R = LSurvivalResp(enter, exit, y, wts, id)
    res = M(R)

    return fit!(res; fitargs...)
end



"""
$DOC_FIT_AJSURV
"""
function fit(
    ::Type{M},
    enter::Vector{<:Real},
    exit::Vector{<:Real},
    y::Y;
    wts::Vector{<:Real} = similar(enter, 0),
    id::Vector{<:AbstractLSurvivalID} = [ID(i) for i in eachindex(y)],
    offset::Vector{<:Real} = similar(enter, 0),
    fitargs...,
) where {M<:AJSurv,Y<:Union{Vector{<:Real},BitVector}}

    R = LSurvivalCompResp(enter, exit, y, wts, id)
    res = M(R)

    return fit!(res; fitargs...)
end

"""
$DOC_FIT_KMSURV
"""
kaplan_meier(enter, exit, y, args...; kwargs...) =
    fit(KMSurv, enter, exit, y, args...; kwargs...)

"""
$DOC_FIT_AJSURV
"""
aalen_johansen(enter, exit, y, args...; kwargs...) =
    fit(AJSurv, enter, exit, y, args...; kwargs...)

##################################################################################################################### 
# Summary functions for non-parametric survival models
#####################################################################################################################
function StatsBase.isfitted(m::M) where {M<:KMSurv}
    m.fit
end

"""
$DOC_VARIANCE_KMSURV
"""
function StatsBase.stderror(m::KMSurv)
    var = m.surv .* m.surv .* cumsum(m.events ./ (m.riskset .* (m.riskset .- m.events)))
    sqrt.(var)
end


function confint_normal(m::KMSurv; level = 0.95)
    se = stderror(m)
    halfalpha = (1.0 - level) / 2.0
    zcrit = quantile.(Normal(), [halfalpha, 1.0 - halfalpha])
    cimat = reduce(hcat, [m.surv .+ zcriti .* se for zcriti in zcrit])
    cimat
end

function confint_lognlog(m::KMSurv; level = 0.95)
    se = stderror(m)
    halfalpha = (1.0 - level) / 2.0
    zcrit = quantile.(Normal(), [halfalpha, 1.0 - halfalpha])
    logci = reduce(
        hcat,
        [
            log.(m.surv) .* exp.(zcriti .* se ./ (m.surv .* log.(m.surv))) for
            zcriti in zcrit
        ],
    )
    cimat = exp.(logci)
    cimat
end

"""
$DOC_VARIANCE_AJSURV
"""
function StatsBase.stderror(m::AJSurv)
    riski = m.risk[:, 1]
    d = sum(m.events, dims = 2)
    sm1 = vcat(1.0, m.surv[1:end-1])
    vv = [
        (
            cumsum(
                (m.surv .* m.risk[:, j]) .* (m.surv .* m.risk[:, j]) .*
                (m.riskset .- 1.0) .* m.riskset .^ (-3.0) .* d,
                dims = 1,
            ) + cumsum(
                (sm1) .* (sm1) .* (1.0 .- 2.0 .* m.risk[:, j]) .* (m.riskset .- 1.0) .*
                m.riskset .^ (-3.0) .* m.events[:, j],
                dims = 1,
            )
        ) for j = 1:size(m.risk, 2)
    ]

    var = reduce(hcat, vv)
    sqrt.(var)
end


function confint_normal(m::AJSurv; level = 0.95)
    se = stderror(m)
    halfalpha = (1.0 - level) / 2.0
    zcrit = quantile.(Normal(), [halfalpha, 1.0 - halfalpha])
    cimat = reduce(
        hcat,
        [
            reduce(hcat, [m.risk[:, j] .+ zcriti .* se[:, j] for zcriti in zcrit]) for
            j = 1:size(m.events, 2)
        ],
    )
    cimat
end


"""
$DOC_VARIANCE_KMSURV
"""
function StatsBase.confint(m::KMSurv; level = 0.95, method = "normal")
    method == "lognlog" ? confint_lognlog(m, level = level) :
    confint_normal(m, level = level)
end

"""
$DOC_VARIANCE_AJSURV
"""
function StatsBase.confint(m::AJSurv; level = 0.95, method = "normal")
    method != "normal" ? @warn("only method=normal CI is implemented for AJSurv objects") :
    confint_normal(m, level = level)
end

function StatsBase.isfitted(m::M) where {M<:AJSurv}
    m.fit
end


function Base.show(io::IO, m::M; maxrows = 20) where {M<:KMSurv}
    if !m.fit
        println(io, "Model not yet fitted")
        return nothing
    end
    resmat = hcat(m.times, m.surv, m.events, m.riskset)
    head = ["time", "survival", "# events", "at risk"]
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
    str = """\nKaplan-Meier Survival\n"""
    str *= String(take!(iob))
    str *= "Number of events: $(@sprintf("%8g", sum(m.events)))\n"
    str *= "Number of unique event times: $(@sprintf("%8g", length(m.events)))\n"
    println(io, str)
end


function Base.show(io::IO, m::M; maxrows = 20) where {M<:AJSurv}
    if !m.fit
        println(io, "Model not yet fitted")
        return nothing
    end
    types = m.R.eventtypes[2:end]
    ev = ["# events (j=$jidx)" for (jidx, j) in enumerate(types)]
    rr = ["risk (j=$jidx)" for (jidx, j) in enumerate(types)]

    resmat = hcat(m.times, m.surv, m.events, m.riskset, m.risk)
    head = ["time", "survival", ev..., "at risk", rr...]
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
    str = """\nKaplan-Meier Survival, Aalen-Johansen risk\n"""
    str *= String(take!(iob))
    for (jidx, j) in enumerate(types)
        str *= "Number of events (j=$j): $(@sprintf("%8g", sum(m.events[:,jidx])))\n"
    end
    str *= "Number of unique event times: $(@sprintf("%8g", length(m.events[:,1])))\n"
    println(io, str)
end

Base.show(m::M; kwargs...) where {M<:AJSurv} =
    Base.show(stdout, m::M; kwargs...) where {M<:AJSurv}
Base.show(m::M; kwargs...) where {M<:KMSurv} =
    Base.show(stdout, m::M; kwargs...) where {M<:KMSurv};



