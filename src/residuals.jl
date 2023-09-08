######################################################################
# residuals from fitted Cox models
######################################################################
"""
$DOC_RESIDUALS
"""
function StatsBase.residuals(m::M; type = "martingale") where {M<:PHModel}
    valid_methods = ["schoenfeld", "score", "martingale"]
    whichmethod = findall(valid_methods .== lowercase(type))
    thismethod = valid_methods[whichmethod][1]
    if thismethod == "martingale"
        resid = resid_martingale(m)
    elseif thismethod == "score"
        resid = resid_score(m)
    elseif thismethod == "schoenfeld"
        resid = resid_schoenfeld(m)
    else
        throw("Method $type not supported yet")
    end
    return resid
end

"""
######################################################################
```julia
dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
)
ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow", maxiter=0)
resid_martingale(ft)
r = exp(ft.P._B[1])
dat1.status .- [r/(3r+3), r/(3r+3), r/(3r+3) + 2r/(r+3), 1/(3r+3) + 2/(r+3), 1/(3r+3) + 2/(r+3), 1/(3r+3) + 2/(r+3) + 1]

ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow")
resid_martingale(ft)
r = exp(ft.P._B[1])
dat1.status .- [r/(3r+3), r/(3r+3), r/(3r+3) + 2r/(r+3), 1/(3r+3) + 2/(r+3), 1/(3r+3) + 2/(r+3), 1/(3r+3) + 2/(r+3) + 1]


ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="efron", maxiter=0)
resid_martingale(ft)
r = exp(ft.P._B[1])
dat1.status .- [r/(3r+3), r/(3r+3), r/(3r+3) + r/(r+3) + r/(r+5), 1/(3r+3) + 1/(r+3) + 1/(r+5), 1/(3r+3) + 1/(r+3) + 2/(r+5), 1/(3r+3) + 1/(r+3) + 2/(r+5) + 1]

ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="efron")
resid_martingale(ft)
r = exp(ft.P._B[1])
dat1.status .- [r/(3r+3), r/(3r+3), r/(3r+3) + r/(r+3) + r/(r+5), 1/(3r+3) + 1/(r+3) + 1/(r+5), 1/(3r+3) + 1/(r+3) + 2/(r+5), 1/(3r+3) + 1/(r+3) + 2/(r+5) + 1]


######################################################################
dat2 = (
    enter = [1,2,5,2,1,7,3,4,8,8],
    exit = [2,3,6,7,8,9,9,9,14,17],
    status = [1,1,1,1,1,1,1,0,0,0],
    x = [1,0,0,1,0,1,1,1,0,0]
)

ft = coxph(@formula(Surv(enter, exit,status)~x),dat2, keepx=true, keepy=true, ties="breslow")
resid_martingale(ft)
[0.521119,0.657411,0.789777,0.247388,-0.606293,0.369025,-0.068766,-1.068766,-0.420447,-0.420447]



ft = coxph(@formula(Surv(enter, exit,status)~x),dat2, keepx=true, keepy=true, ties="efron")




######################################################################
dat3 = (
    time = [1,1,2,2,2,2,3,4,5],
    status = [1,0,1,1,1,0,0,1,0],
    x = [2,0,1,1,0,1,0,1,0],
    wt = [1,2,3,4,3,2,1,2,1]
)


ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="efron", maxiter=0)
resid_martingale(ft)
[1-1/19,0-1/19,1-(1/19+10/48+20/114+10/84),1-(1/19+10/48+20/114+10/84),1-(1/19+10/48+20/114+10/84),0-(1/19+10/48+10/38+10/28),0-(1/19+10/48+10/38+10/28),1-(1/19+10/48+10/38+10/28+2/3),0-(1/19+10/48+10/38+10/28+2/3)]
expected_FH(ft)

ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="efron")
resid_martingale(ft)
[18/19,-1/19,473/1064,473/1064,473/1064,-0.881265664,-0.881265664,-0.547932331,-1.547932331]

ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="breslow", maxiter=0)
resid_martingale(ft)
[18/19,−1/19,49/152,49/152,49/152,−103/152,−103/152,−157/456,−613/456]

ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="breslow")
resid_martingale(ft)
[0.85531,-0.02593,0.17636,0.17636,0.65131,-0.82364,-0.34869,-0.64894,-0.69808]
```
"""
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
