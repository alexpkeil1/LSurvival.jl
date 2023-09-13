function pop(R::T) where {T<:LSurvivalResp}
    uid = unique(R.id)[end]
    idx = findall(getfield.(R.id, :value) .== uid.value)
    nidx = setdiff(collect(eachindex(R.id)),idx)
    Ri = LSurvivalResp(R.enter[idx], R.exit[idx], R.y[idx], R.wts[idx], R.id[idx]; origintime= R.origin)
    Rj = LSurvivalResp(R.enter[nidx], R.exit[nidx], R.y[nidx], R.wts[nidx], R.id[nidx],origintime= R.origin)
    Ri, Rj, idx, nidx
end

function popat!(P::T, idxi, idxj) where {T<:PHParms}
    Pi = PHParms(P.X[idxi,:], P._B, P._r[idxi], P._LL, P._grad, P._hess, 1, P.p)
    P.X =  P.X[idxj,:]
    P._r = P._r[idxj]
    P.n -= length(idxi)
    Pi
end

function push(Ri::T, Rj::T) where {T<:LSurvivalResp}
    Ri = LSurvivalResp(
        vcat(Ri.enter, Rj.enter), 
        vcat(Ri.exit, Rj.exit), 
        vcat(Ri.y, Rj.y), 
        vcat(Ri.wts, Rj.wts), 
        vcat(Ri.id, Rj.id); 
        origintime = min(Ri.origin, Rj.origin))
end

function push!(Pi::T, Pj::T) where {T<:PHParms}
    Pj.X =  vcat(Pi.X, Pj.X)
    Pj._r = vcat(Pi._r, Pj._r)
    Pj.n +=1
    nothing
end


"""
$DOC_JACKKNIFE
"""
function jackknife(m::M) where {M<:PHModel}
    uid = unique(m.R.id)
    coefs = zeros(length(uid), length(coef(m)))
    R = deepcopy(m.R)
    P = deepcopy(m.P)
    for i in eachindex(uid)
        Ri, Rj, idxi, idxj = pop(m.R);
        Pi = popat!(m.P, idxi, idxj)
        mi= PHModel(Rj, m.P, m.formula, m.ties, false, m.bh[1:length(Rj.eventtimes),:], nothing)
        fit!(mi, getbasehaz=false)
        coefs[i,:] = coef(mi)
        m.R = push(Ri, Rj)
        push!(Pi, m.P)
    end
    m.R, m.P = R, P
    coefs
end


function jackknife_vcov(m::M) where {M<:PHModel}
    N = nobs(m)
    #comparing estimate with jackknife estimate with bootstrap mean
    jk = jackknife(m);
    covjk = cov(jk, corrected=false) .* (N-1)
    covjk
end

