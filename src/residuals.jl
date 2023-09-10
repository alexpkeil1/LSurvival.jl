######################################################################
# residuals from fitted Cox models
######################################################################
"""
$DOC_RESIDUALS
"""
function StatsBase.residuals(m::M; type="martingale") where {M<:PHModel}
    valid_methods = ["schoenfeld", "score", "martingale", "dfbeta", "scaled_schoenfeld"]
    whichmethod = findall(valid_methods .== lowercase(type))
    thismethod = valid_methods[whichmethod][1]
    if thismethod == "martingale"
        resid = resid_martingale(m)
    elseif thismethod == "score"
        resid = resid_score(m)
    elseif thismethod == "schoenfeld"
        resid = resid_schoenfeld(m)
    elseif thismethod == "dfbeta"
        resid = resid_dfbeta(m)
    elseif thismethod == "scaled_schoenfeld"
        resid = resid_schoenfeld(m) * inv(m.P._hess)
    else
        throw("Method $type not supported yet")
    end
    return resid
end

function resid_martingale(m::M) where {M<:PHModel}
    Nw = Float64.(m.R.y .> 0.0)
    if m.ties == "breslow"
        resid = Nw .- expected_NA(m)
    elseif m.ties == "efron"
        resid = Nw .- expected_FH(m)
    else
        throw("Ties method not recognized")
    end
    resid
end

function expected_NA(m::M) where {M<:PHModel}
    # Nelson-Aalen-Breslow
    rr = m.P._r
    dΛ0 = m.bh[:, 1]
    bht = m.bh[:, 4]
    whichbhindex = [
        findall((bht .<= m.R.exit[i]) .&& (bht .> m.R.enter[i])) for
        i in eachindex(m.R.exit)
    ]
    E = [sum(rr[i] .* dΛ0[whichbhindex[i]]) for i in eachindex(whichbhindex)]
    return E
end

function expected_FH(m::M) where {M<:PHModel}
    # Fleming-Harrington-Efron (Nonparametric estimation of the survival distribution in censored data, 1984)
    rr = m.P._r
    eventtimes = m.R.eventtimes
    whichbhindex = [
        findall((eventtimes .<= m.R.exit[i]) .&& (eventtimes .> m.R.enter[i])) for
        i in eachindex(m.R.exit)
    ]
    whichbhcaseindex = [
        findall(isapprox.(eventtimes, m.R.exit[i]) .&& (m.R.y[i] > 0)) for
        i in eachindex(m.R.exit)
    ]
    E0r, E0c = expected_efronbasehaz(m)
    E = [
        sum(
            rr[i] .* vcat(
                E0r[setdiff(whichbhindex[i], whichbhcaseindex[i])],
                E0c[whichbhcaseindex[i]],
            ),
        ) for i in eachindex(whichbhindex)
    ]
    return E
end


function expected_efronbasehaz(m::M) where {M<:PHModel}
    ne = length(m.R.eventtimes)
    denr, denc, _sumwtriskset, _sumwtcase =
        zeros(Float64, ne), zeros(Float64, ne), zeros(Float64, ne), zeros(Float64, ne)
    @inbounds @simd for j = 1:ne
        _outj = m.R.eventtimes[j]
        risksetidx = findall((m.R.enter .< _outj) .&& (m.R.exit .>= _outj))
        caseidx =
            findall((m.R.y .> 0) .&& isapprox.(m.R.exit, _outj) .&& (m.R.enter .< _outj))
        nties = length(caseidx)
        effwts = efron_weights(nties)
        denj = expected_denj(m.P._r, m.R.wts, caseidx, risksetidx, nties, j)
        _sumwtriskset[j] = sum(m.R.wts[risksetidx])
        _sumwtcase[j] = sum(m.R.wts[caseidx])
        denr[j] = sum(denj) # correct
        denc[j] = sum(denj .* (1.0 .- effwts))
    end
    denr, denc
end


function expected_denj(_r, wts, caseidx, risksetidx, nties, j)
    # expected value denominator for all observations at a given time 
    _rcases = view(_r, caseidx)
    _rriskset = view(_r, risksetidx)
    _wtcases = view(wts, caseidx)
    _wtriskset = view(wts, risksetidx)
    #
    risksetrisk = sum(_wtriskset .* _rriskset)
    #
    effwts = efron_weights(nties)
    sw = sum(_wtcases)
    aw = sw / nties
    casesrisk = sum(_wtcases .* _rcases)
    dens = [risksetrisk - casesrisk * ew for ew in effwts]
    aw ./ dens # using Efron estimator
end

function resid_score(m::M) where {M<:PHModel}
    L = resid_Lmat(m)
    resids = [sum(Lmat, dims=2)[:] for Lmat in L]
    reduce(hcat, resids)
end


function resid_schoenfeld(m::M) where {M<:PHModel}
    L = resid_Lmat(m)
    resids = [sum(Lmat, dims=1)[:] for Lmat in L]
    reduce(hcat, resids)
end

function resid_dfbeta(m::M) where {M<:PHModel}
    L = resid_score(m)
    H = m.P._hess
    dfbeta = .- L * inv(H)
    return dfbeta
end

"""
$DOC_ROBUST_VCOV
"""
function robust_vcov(m::M) where {M<:PHModel}
    dfbeta = resid_dfbeta(m)
    robVar = dfbeta'dfbeta
    return robVar
end


function resid_Lmat(m::M) where {M<:PHModel}
    if !isnothing(m.RL)
        return m.RL
    end
    maxties = maximum(m.bh[:, 3])
    if (m.ties == "breslow") || (maxties <= 1.0) && (m.ties == "efron")
        L = resid_Lmat_breslow(m)
    elseif m.ties == "efron"
        L = resid_Lmat_efron(m)
    else
        throw("Ties method not recognized")
    end
    L
end

function resid_Lmat_breslow(m::M) where {M<:PHModel}
    X = m.P.X
    nxcols = size(X, 2)
    nobs = size(X, 1)
    ntimes = length(m.R.eventtimes)
    #wts = m.R.wts
    Nw = Float64.(m.R.y .> 0.0)
    #maxties = maximum(m.bh[:, 3])
    dM, dt, di = dexpected_NA(m)
    muX = muX_t(m, di)
    #
    dM .*= -1
    for i in eachindex(dM)
        dM[i][end] += Nw[i]
    end
    L = [zeros(nobs, ntimes) for nx = 1:nxcols]
    for j = 1:nxcols
        for i = 1:nobs
            pr = (X[i, j] .- muX[di[i]]) .* dM[i]
            L[j][i, di[i]] .= pr
        end
    end
    m.RL = L
    L
end

function resid_Lmat_efron(m::M) where {M<:PHModel}
    @warn("not yet working")
    X = m.P.X
    y = m.R.y
    exit = m.R.exit
    nxcols = size(X, 2)
    nobs = size(X, 1)
    ties = m.bh[:, 6]
    times = m.R.eventtimes
    ntimes = length(times)
    Nw = Float64.(y .> 0.0)
    tiedis = [
        ties[findall((m.R.exit[yi] .== times) .&& (Nw[yi] .> 0))] for
        yi in eachindex(m.R.exit)
    ]
    tiesi = [length(t) > 0 ? t[1] : 1.0 for t in tiedis]
    #maxties = maximum(ties)
    #

    dMt, dt, di = dexpected_FH(m)
    muXt = muX_tE(m, di)
    #muXt[1][end] = [.6, .5]

    dMt .*= -1
    for i in eachindex(dMt)
        dMt[i][end] .+= Nw[i] ./ tiesi[i]
    end
    L = [zeros(nobs, ntimes) for nx = 1:nxcols]
    for j = 1:nxcols
        for i = 1:nobs
            for d in di[i]  # time index of all times
                if (exit[i] == times[d]) && (y[i] > 0)
                    div = ties[d]
                    ew = LSurvival.efron_weights(div)
                else
                    div = 1.0
                    ew = 0.0
                end
                dmidx = findall(di[i] .== d) # map from individual times to all times
                pr = (X[i, j] .- muXt[j][d]) .* (reduce(vcat, dMt[i][dmidx]))
                #pr = (X[i, j] .- sum(muXt[j][d])./ div) .+ sum((X[i, j] .- muXt[j][d]) .* dMt[i][d] .* (1 .- ew))
                L[j][i, d] = sum(pr)
            end
        end
    end
    m.RL = L
    L
end

# goal: 0.243671
#x = 0
#dmt = [0.2, 0.25]
# mux = [0.6, 0.6] # same
#0.2* (x-0.6) + 0.25* (x-0.6)




function muX_t(m::M, whichbhindex) where {M<:PHModel}
    # Nelson-Aalen-Breslow
    X = m.P.X
    bht = m.R.eventtimes
    wts = m.R.wts
    r = m.P._r
    # which event times is each observation at risk?
    nxcols = size(X, 2)
    sumX = zeros(length(bht), nxcols)
    nX = zeros(length(bht), nxcols)
    for j = 1:nxcols
        for i in eachindex(whichbhindex)
            sumX[whichbhindex[i], j] .+= X[i, j] * wts[i] * r[i]
            nX[whichbhindex[i], j] .+= wts[i] * r[i]
        end
    end
    return sumX ./ nX
end


function muX_tE(m::M, whichbhindex) where {M<:PHModel}
    # Fleming-Harrington-Efron
    X = m.P.X
    y = m.R.y
    exit = m.R.exit
    bht = m.R.eventtimes
    wts = m.R.wts
    r = m.P._r
    nties = m.bh[:, 6]
    nxcols = size(X, 2)

    muXE = fill([zeros(Int(j)) for j in nties], nxcols)
    nX = fill([zeros(Int(j)) for j in nties], nxcols)
    for j = 1:nxcols
        for i in eachindex(whichbhindex)
            ties = nties[whichbhindex[i]]    # number of ties at each time index
            untied = findall(ties .<= 1.0)   # time index of untied times
            tied = findall(ties .> 1.0)      # time index of tied times
            #
            for u in untied
                muXE[j][whichbhindex[i][u]] .+= X[i, j] * wts[i] * r[i]
                nX[j][whichbhindex[i][u]] .+= wts[i] * r[i]
            end
            for t in tied
                if (y[i] > 0) && (exit[i] == bht[whichbhindex[i][t]])
                    ew = LSurvival.efron_weights(ties[t])
                    muXE[j][whichbhindex[i][t]] .+= (1 .- ew) .* X[i, j] * wts[i] * r[i]
                    nX[j][whichbhindex[i][t]] .+= (1 .- ew) .* wts[i] * r[i]
                else
                    muXE[j][whichbhindex[i][t]] .+= X[i, j] * wts[i] * r[i]
                    nX[j][whichbhindex[i][t]] .+= wts[i] * r[i]
                end
            end
        end
    end
    for j = 1:nxcols
        for v in eachindex(muXE[j])
            muXE[j][v] ./= nX[j][v]
        end
    end
    muXE
end

function dexpected_NA(m::M) where {M<:PHModel}
    # Nelson-Aalen-Breslow
    rr = m.P._r
    dΛ0 = m.bh[:, 1]
    eventtimes = m.R.eventtimes
    whichbhindex = [
        findall((eventtimes .<= m.R.exit[i]) .&& (eventtimes .> m.R.enter[i])) for
        i in eachindex(m.R.exit)
    ]
    dE = [(rr[i] .* dΛ0[whichbhindex[i]]) for i in eachindex(whichbhindex)]
    dt = [(eventtimes[whichbhindex[i]]) for i in eachindex(whichbhindex)]
    return dE, dt, whichbhindex
end

function dexpected_FH(m::M) where {M<:PHModel}
    # Fleming-Harrington-Efron (Nonparametric estimation of the survival distribution in censored data, 1984)
    rr = m.P._r
    eventtimes = m.R.eventtimes
    whichbhindex = [
        findall((eventtimes .<= m.R.exit[i]) .&& (eventtimes .> m.R.enter[i])) for
        i in eachindex(m.R.exit)
    ]
    whichbhcaseindex = [
        findall(isapprox.(eventtimes, m.R.exit[i]) .&& (m.R.y[i] > 0)) for
        i in eachindex(m.R.exit)
    ]
    dER, dE0 = dexpected_efronbasehaz(m)
    dE = [rr[i] .* dER[whichbhindex[i]] for i in eachindex(whichbhindex)]
    for i in eachindex(whichbhindex)
        if length(whichbhcaseindex[i]) > 0
            xix = findall(whichbhindex[i] .== whichbhcaseindex[i])
            dE[i][xix] = rr[i] .* dE0[whichbhcaseindex[i]]
        end
    end
    dt = [(eventtimes[whichbhindex[i]]) for i in eachindex(whichbhindex)]
    return dE, dt, whichbhindex
end


function dexpected_efronbasehaz(m::M) where {M<:PHModel}
    ne = length(m.R.eventtimes)
    nties = m.bh[:, 6]
    denr, denc, _sumwtriskset, _sumwtcase = [zeros(Int(j)) for j in nties],
    [zeros(Int(j)) for j in nties],
    zeros(Float64, ne),
    zeros(Float64, ne)
    @inbounds @simd for j = 1:ne
        _outj = m.R.eventtimes[j]
        risksetidx = findall((m.R.enter .< _outj) .&& (m.R.exit .>= _outj))
        caseidx =
            findall((m.R.y .> 0) .&& isapprox.(m.R.exit, _outj) .&& (m.R.enter .< _outj))
        nties = length(caseidx)
        effwts = LSurvival.efron_weights(nties)
        denj = LSurvival.expected_denj(m.P._r, m.R.wts, caseidx, risksetidx, nties, j)
        _sumwtriskset[j] = sum(m.R.wts[risksetidx])
        _sumwtcase[j] = sum(m.R.wts[caseidx])
        denr[j] .= denj # correct
        denc[j] .= (denj .* (1.0 .- effwts))
    end
    denr, denc
end