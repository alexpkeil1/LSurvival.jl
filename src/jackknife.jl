function pop(R::T) where {T<:LSurvivalResp}
    uid = unique(R.id)[end]
    idx = findall(getfield.(R.id, :value) .== uid.value)
    nidx = setdiff(collect(eachindex(R.id)), idx)
    Ri = LSurvivalResp(
        R.enter[idx],
        R.exit[idx],
        R.y[idx],
        R.wts[idx],
        R.id[idx];
        origintime = R.origin,
    )
    Rj = LSurvivalResp(
        R.enter[nidx],
        R.exit[nidx],
        R.y[nidx],
        R.wts[nidx],
        R.id[nidx],
        origintime = R.origin,
    )
    Ri, Rj, idx, nidx
end

function pop(R::T) where {T<:LSurvivalCompResp}
    uid = unique(R.id)[end]
    idx = findall(getfield.(R.id, :value) .== uid.value)
    nidx = setdiff(collect(eachindex(R.id)), idx)
    Ri = LSurvivalCompResp(
        R.enter[idx],
        R.exit[idx],
        R.y[idx],
        R.wts[idx],
        R.id[idx];
        origintime = R.origin,
        etypes = R.eventtypes,
        ematrix = R.eventmatrix[idx,:]
    )
    Rj = LSurvivalCompResp(
        R.enter[nidx],
        R.exit[nidx],
        R.y[nidx],
        R.wts[nidx],
        R.id[nidx],
        origintime = R.origin,
        etypes = R.eventtypes,
        ematrix = R.eventmatrix[nidx,:]
    )
    Ri, Rj, idx, nidx
end


function popat!(P::T, idxi, idxj) where {T<:PHParms}
    Pi = PHParms(P.X[idxi, :], P._B, P._r[idxi], P._LL, P._grad, P._hess, 1, P.p)
    P.X = P.X[idxj, :]
    P._r = P._r[idxj]
    P.n -= length(idxi)
    Pi
end

function push(Ri::T, Rj::T) where {T<:LSurvivalResp}
    LSurvivalResp(
        vcat(Ri.enter, Rj.enter),
        vcat(Ri.exit, Rj.exit),
        vcat(Ri.y, Rj.y),
        vcat(Ri.wts, Rj.wts),
        vcat(Ri.id, Rj.id);
        origintime = min(Ri.origin, Rj.origin),
    )
end

function push(Ri::T, Rj::T) where {T<:LSurvivalCompResp}
    LSurvivalCompResp(
        vcat(Ri.enter, Rj.enter),
        vcat(Ri.exit, Rj.exit),
        vcat(Ri.y, Rj.y),
        vcat(Ri.wts, Rj.wts),
        vcat(Ri.id, Rj.id);
        origintime = min(Ri.origin, Rj.origin),
        etypes = sort(unique(vcat(Ri.eventtypes, Rj.eventtypes))),
        ematrix = vcat(Ri.eventmatrix, Rj.eventmatrix)
    )
end

function push!(Pi::T, Pj::T) where {T<:PHParms}
    Pj.X = vcat(Pi.X, Pj.X)
    Pj._r = vcat(Pi._r, Pj._r)
    Pj.n += 1
    nothing
end


"""
$DOC_JACKKNIFE
"""
function jackknife(m::M; kwargs...) where {M<:PHModel}
    uid = unique(m.R.id)
    coefs = zeros(length(uid), length(coef(m)))
    R = deepcopy(m.R)
    P = deepcopy(m.P)
    for i in eachindex(uid)
        Ri, Rj, idxi, idxj = pop(m.R)
        Pi = popat!(m.P, idxi, idxj)
        mi = PHModel(
            Rj,
            m.P,
            m.formula,
            m.ties,
            false,
            m.bh[1:length(Rj.eventtimes), :],
            nothing,
        )
        fit!(mi, getbasehaz = false; kwargs...)
        coefs[i, :] = coef(mi)
        m.R = push(Ri, Rj)
        push!(Pi, m.P)
    end
    m.R, m.P = R, P
    coefs
end


function jackknife_vcov(m::M) where {M<:PHModel}
    N = nobs(m)
    #comparing estimate with jackknife estimate with bootstrap mean
    jk = jackknife(m)
    covjk = cov(jk, corrected = false) .* (N - 1)
    covjk
end


"""
Obtain jackknife (leave-one-out) estimates from a Kaplan-Meier survival curve (survival at end of follow-up) by refitting the model n times


## Signatures

```julia
jackknife(m::M;kwargs...) where {M<:KMSurv}
```

```@example
using LSurvival, Random, StatsBase

dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])

dat1clust = (
  id = [1, 2, 3, 3, 4, 4, 5, 5, 6, 6],
  enter = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
  exit = [1, 1, 1, 6, 1, 6, 1, 8, 1, 9],
  status = [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
  x = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
)

m = kaplan_meier(dat1.time, dat1.status)
a = aalen_johansen(dat1.time, dat1.status)
mc = kaplan_meier(dat1clust.enter, dat1clust.exit, dat1clust.status, id=ID.(dat1clust.id))
ac = aalen_johansen(dat1clust.enter, dat1clust.exit, dat1clust.status, id=ID.(dat1clust.id))
jk = jackknife(m);
jkc = jackknife(mc);
jka = jackknife(a);
bs = bootstrap(mc, 100);
std(bs[:,1])
stderror(m, type="jackknife")
stderror(mc, type="jackknife")
@assert jk == jkc
```
"""
function jackknife(m::M; kwargs...) where {M<:KMSurv}
    uid = unique(m.R.id)
    res = zeros(length(uid))
    R = deepcopy(m.R)
    for i in eachindex(uid)
        Ri, Rj, _, _ = pop(m.R)
        mi = KMSurv(Rj)
        fit!(mi; kwargs...)
        res[i, :] = mi.surv[end:end]
        m.R = push(Ri, Rj)
    end
    m.R = R
    res
end

"""
Obtain jackknife (leave-one-out) estimates from a Aalen-Johansen risk curve (risk at end of follow-up) by refitting the model n times

## Signatures

```julia
jackknife(m::M;kwargs...) where {M<:AJSurv}
```

"""
function jackknife(m::M; kwargs...) where {M<:AJSurv}
    uid = unique(m.R.id)
    res = zeros(length(uid))
    R = deepcopy(m.R)
    for i in eachindex(uid)
        Ri, Rj, _, _ = pop(m.R)
        mi = AJSurv(Rj)
        fit!(mi; kwargs...)
        res[i, :] = mi.surv[end:end]
        m.R = push(Ri, Rj)
    end
    m.R = R
    res
end
