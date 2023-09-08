# to implement
# - robust standard error estimate for Cox model
# - using formulas

# currently working here:
# Breslow: score, schoenfeld residuals

# currently NOT working here:
# Efron: score, schoenfeld residuals


using LSurvival, Random, Optim, BenchmarkTools

######################################################################
# residuals from fitted Cox models
######################################################################



"""

Score residuals: Per observation contribution to score function 

```julia
using LSurvival

dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
)
ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow")



X = ft.P.X
M = residuals(ft, type="martingale")
r = exp(ft.P._B[1])
# nXp matrix used for schoenfeld and score residuals
truthmat = permutedims(hcat(
  [(1-r/(r+1)) * (1-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), (1-r/(r+3)) * (1-2r/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (1-2/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-2/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-2/(r+3)), (0-0) * (1-1)]
))
truth = sum(truthmat, dims=2)[:]
S = resid_score(ft)[:]
@assert all(isapprox.(S, truth))
# assertions for testing breslow ties estimates under convergence for dat1
if length(ft.P._LL)>1
    @assert isapprox(ft.P._B[1], 1.475285, atol=0.000001)
    @assert isapprox(ft.P._LL[[1,end]], [-4.56434819, -3.82474951], atol=0.000001)
    @assert isapprox(-ft.P._hess[1], 0.6341681, atol=0.000001)
end

ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="efron", maxiter=0)
S = resid_score(ft)[:]

truthmat = permutedims(hcat(
  [(1-r/(r+1)) * (1-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), (1-r/(r+3)) * (1-r/(r+3)) + (1-r/(r+5)) * (1-2r/(r+5))/2, 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (1-1/(r+3)) +  (0-r/(r+5)) * (1-2/(r+5))/2, 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-1/(r+3)) +  (0-r/(r+5)) * (0-2/(r+5)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-1/(r+3)) +  (0-r/(r+5)) * (0-2/(r+5)), (0-0) * (1-1)]
))
truth = sum(truthmat, dims=2)[:]


[5/12, -1/12, 55/155, -5/144, 29/144, 29/144]
S
```
"""
function resid_score(m::M) where {M<:PHModel}
    L = resid_Lmat(m)
    resids = [sum(Lmat, dims = 2)[:] for Lmat in L]
    reduce(hcat, resids)
end


"""
Schoenfeld residuals: Per time contribution to score function 
```julia
using LSurvival
dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
)
ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow", maxiter=0)


X = ft.P.X
M = residuals(ft, type="martingale")
S = resid_schoenfeld(ft)[:]
r = exp(ft.P._B[1])
truthmat = permutedims(hcat(
  [(1-r/(r+1)) * (1-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), (1-r/(r+3)) * (1-2r/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (1-2/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-2/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-2/(r+3)), (0-0) * (1-1)]
))
truth = sum(truthmat, dims=1)[:]
all(isapprox.(S, truth))

ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow")

X = ft.P.X
S = resid_schoenfeld(ft)[:]
r = exp(ft.P._B[1])
truthmat = permutedims(hcat(
  [(1-r/(r+1)) * (1-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), (1-r/(r+3)) * (1-2r/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (1-2/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-2/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-2/(r+3)), (0-0) * (1-1)]
))
truth = sum(truthmat, dims=1)[:]
all(isapprox.(S, truth))


ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="efron", maxiter=0)
S = resid_schoenfeld(ft)


# df beta residuals
ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow")
L = resid_score(ft) # n X p
H = ft.P._hess   # p X p
dfbeta = L*inv(H)
robVar = dfbeta'dfbeta
sqrt(robVar)


```
"""
function resid_schoenfeld(m::M) where {M<:PHModel}
    L = resid_Lmat(m)
    resids = [sum(Lmat, dims = 1)[:] for Lmat in L]
    reduce(hcat, resids)
end

function resid_dfbeta(m::M) where {M<:PHModel}
    L = resid_score(ft)
    H = ft.P._hess 
    dfbeta = L*inv(H)
    return dfbeta
end

function robust_vcov(m::M) where {M<:PHModel}
    dfbeta = resid_dfbeta(ft)
    robVar = dfbeta'dfbeta 
    return robVar
end



"""
ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow", maxiter=0)

X = ft.P.X
L = resid_Lmat(ft)[1]
r = exp(ft.P._B[1])
truthmat = permutedims(hcat(
  [(1-r/(r+1)) * (1-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), (1-r/(r+3)) * (1-2r/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (1-2/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-2/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-2/(r+3)), (0-0) * (1-1)]
))

@assert sum((L .- truthmat).^2)<0.001


ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="efron", maxiter=0)

ft.ties = "breslow"
X = ft.P.X
L = resid_Lmat(ft)[1]
r = exp(ft.P._B[1])
truthmat = permutedims(hcat(
  [(1-r/(r+1)) * (1-r/(3r+3)), 0                                                       , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), 0                                                       , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), (1-r/(r+3)) * (1-r/(r+3)) + (1-r/(r+5)) * (1-2r/(r+5))/2, 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (1-1/(r+3)) +  (0-r/(r+5)) * (1-2/(r+5))/2, 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-1/(r+3)) +  (0-r/(r+5)) * (0-2/(r+5)),   0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-1/(r+3)) +  (0-r/(r+5)) * (0-2/(r+5)), (0-0) * (1-1)]
))
sum(truthmat, dims=2)
# 3: 0.3819444444444444, 4: -0.03472222
sum(L, dims=2)

L
truthmat
@assert sum((L .- truthmat).^2)<0.001


"""
function resid_Lmat(m::M) where {M<:PHModel}
    if !isnothing(m.RL) 
        return m.RL
    end
    X = m.P.X
    nxcols = size(X, 2)
    nobs = size(X, 1)
    ntimes = length(m.R.eventtimes)
    Nw = Float64.(m.R.y .> 0.0)
    maxties = maximum(m.bh[:, 3])
    if (m.ties == "breslow") || (maxties <= 1.0) && (m.ties == "efron")
        dM, dt, di = dexpected_NA(m)
        muX = muX_t(m, di)
    elseif m.ties == "efron"
        @warn("not yet working")
        dM, dt, di = dexpected_FH(m)
        muX, muXc = muX_t2(m, di)
    else
        throw("Ties method not recognized")
    end
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

function muX_t2(m::M, whichbhindex) where {M<:PHModel}
    # Fleming-Harrington-Efron
    X = m.P.X
    y = m.R.y
    bht = m.R.eventtimes
    wts = m.R.wts
    r = m.P._r
    nties = m.bh[:, 3]
    # which event times is each observation at risk?
    nxcols = size(X, 2)
    sumX = zeros(length(bht), nxcols)
    nX = zeros(length(bht), nxcols)
    sumXcases = zeros(length(bht), nxcols)
    nXcases = zeros(length(bht), nxcols)
    for j = 1:nxcols
        for i in eachindex(whichbhindex)
            ties = nties[whichbhindex[i]]
            sumX[whichbhindex[i], j] .+= X[i, j] * wts[i] * r[i]
            nX[whichbhindex[i], j] .+= wts[i] * r[i]
            sumXcases[whichbhindex[i], j] .+= X[i, j] * wts[i] * r[i]
            nXcases[whichbhindex[i], j] .+= wts[i] * r[i]
            if (y[i] > 0.0) && (ties[end] .> 1)
                ew = LSurvival.efron_weights.(ties[end])
                sumXcases[whichbhindex[i][end], j] -= (X[i, j] * wts[i] * r[i])
                nXcases[whichbhindex[i][end], j]   -= (wts[i] * r[i])
                for nt in 1:Int(ties[end])
                    sumXcases[whichbhindex[i][end], j] += (1-ew[nt])*(X[i, j] * wts[i] * r[i])/ties[end]
                    #sumXcases[whichbhindex[i][end], j] += (X[i, j] * wts[i] * r[i])/ties[end]
                    nXcases[whichbhindex[i][end], j]   += (1-ew[nt])*(wts[i] * r[i])
                end
            end
            #nties[whichbhindex[i]]
            #LSurvival.efron_weights(2)
        end
    end
    return sumX ./ nX, sumXcases ./ nXcases
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
    #E0r, E0c = expected_efronbasehaz(m)
    dE0r, dE0c = LSurvival.expected_efronbasehaz(m)
    dE = [
        rr[i] .* vcat(
            reduce(
                vcat,
                dE0r[setdiff(whichbhindex[i], whichbhcaseindex[i])],
                init = Float64[],
            ),
            reduce(vcat, dE0c[whichbhcaseindex[i]], init = Float64[]),
        ) for i in eachindex(whichbhindex)
    ]
    dt = [(eventtimes[whichbhindex[i]]) for i in eachindex(whichbhindex)]
    return dE, dt, whichbhindex
end



function dexpected_efronbasehaz(m::M) where {M<:PHModel}
    ne = length(m.R.eventtimes)
    denr, denc, _sumwtriskset, _sumwtcase = Array{Array{Float64,1}}(undef, ne),
    Array{Float64,1}(undef, ne),
    zeros(Float64, ne),
    zeros(Float64, ne)
    @inbounds @simd for j = 1:ne
        _outj = m.R.eventtimes[j]
        risksetidx = findall((m.R.enter .< _outj) .&& (m.R.exit .>= _outj))
        caseidx =
            findall((m.R.y .> 0) .&& isapprox.(m.R.exit, _outj) .&& (m.R.enter .< _outj))
        nties = length(caseidx)
        effwts = LSurvival.efron_weights(nties)
        # expected value denominator for all at-risk observations at a given time 
        denj = expected_denj(m.P._r, m.R.wts, caseidx, risksetidx, nties, j)
        _sumwtriskset[j] = sum(m.R.wts[risksetidx])
        _sumwtcase[j] = sum(m.R.wts[caseidx])
        # expected value denominator for all non-cases at a given time 
        denr[j] = denj
        # expected value denominator for all cases at a given time 
        denc[j] = sum(denj .* (1.0 .- effwts))
    end
    denr, denc
end

#=
function expected_denj(_r, wts, caseidx, risksetidx, nties, j)
    # expected value denominator for all observations at a given time 
    _rcases = view(_r, caseidx)
    _rriskset = view(_r, risksetidx)
    _wtcases = view(wts, caseidx)
    _wtriskset = view(wts, risksetidx)
    #
    risksetrisk = sum(_wtriskset .* _rriskset)
    #
    effwts = LSurvival.efron_weights(nties)
    sw = sum(_wtcases)
    aw = sw / nties
    casesrisk = sum(_wtcases .* _rcases)
    dens = [risksetrisk - casesrisk * ew for ew in effwts]
    aw ./ dens # using Efron estimator
end
=#


######################################################################
# fitting with optim
######################################################################

function fit!(
    m::PHModel;
    verbose::Bool = false,
    maxiter::Integer = 500,
    atol::Float64 = 0.0,
    rtol::Float64 = 0.0,
    gtol::Float64 = 1e-8,
    start = nothing,
    keepx = false,
    keepy = false,
    bootstrap_sample = false,
    bootstrap_rng = MersenneTwister(),
    kwargs...,
)
    m = bootstrap_sample ? bootstrap(bootstrap_rng, m) : m
    start = isnothing(start) ? zeros(length(m.P._B)) : start
    if haskey(kwargs, :ties)
        m.ties = kwargs[:ties]
    end
    ne = length(m.R.eventtimes)
    risksetidxs, caseidxs =
        Array{Array{Int,1},1}(undef, ne), Array{Array{Int,1},1}(undef, ne)
    _sumwtriskset, _sumwtcase = zeros(Float64, ne), zeros(Float64, ne)
    @inbounds @simd for j = 1:ne
        _outj = m.R.eventtimes[j]
        fr = findall((m.R.enter .< _outj) .&& (m.R.exit .>= _outj))
        fc = findall((m.R.y .> 0) .&& isapprox.(m.R.exit, _outj) .&& (m.R.enter .< _outj))
        risksetidxs[j] = fr
        caseidxs[j] = fc
        _sumwtriskset[j] = sum(m.R.wts[fr])
        _sumwtcase[j] = sum(m.R.wts[fc])
    end
    # cox risk and set to zero were both in step cox - return them?
    # loop over event times
    #LSurvival._coxrisk!(m.P) # updates all elements of _r as exp(X*_B)
    #LSurvival._settozero!(m.P)
    #LSurvival._partial_LL!(m, risksetidxs, caseidxs, ne, den)

    function coxupdate!(
        F,
        G,
        H,
        beta,
        m;
        ne = ne,
        caseidxs = caseidxs,
        risksetidxs = risksetidxs,
    )
        m.P._LL[1] = isnothing(F) ? m.P._LL[1] : F
        m.P._grad = isnothing(G) ? m.P._grad : G
        m.P._hess = isnothing(H) ? m.P._hess : H
        m.P._B = isnothing(beta) ? m.P._B : beta
        #
        LSurvival._update_PHParms!(m, ne, caseidxs, risksetidxs)
        # turn into a minimization problem
        F = -m.P._LL[1]
        m.P._grad .*= -1.0
        m.P._hess .*= -1.0
        F
    end

    fgh! = TwiceDifferentiable(
        Optim.only_fgh!((F, G, H, beta) -> coxupdate!(F, G, H, beta, m)),
        start,
    )
    opt = NewtonTrustRegion()
    #opt = IPNewton()
    #opt = Newton()
    res = optimize(
        fgh!,
        start,
        opt,
        Optim.Options(
            f_abstol = atol,
            f_reltol = rtol,
            g_tol = gtol,
            iterations = maxiter,
            store_trace = true,
        ),
    )
    verbose && println(res)

    m.fit = true
    m.P._grad .*= -1.0
    m.P._hess .*= -1.0
    m.P._LL = [-x.value for x in res.trace]
    basehaz!(m)
    m.P.X = keepx ? m.P.X : nothing
    m.R = keepy ? m.R : nothing
    m
end



id, int, outt, data = LSurvival.dgm(MersenneTwister(345), 100, 10; afun = LSurvival.int_0);
data[:, 1] = round.(data[:, 1], digits = 3);
d, X = data[:, 4], data[:, 1:3];


# not-yet-fit PH model object
#m = PHModel(R, P, "breslow")
#LSurvival._fit!(m, start = [0.0, 0.0, 0.0], keepx=true, keepy=true)
#isfitted(m)
R = LSurvResp(int, outt, d)
P = PHParms(X)
m = PHModel(R, P);  #default is "efron" method for ties
@btime res = LSurvival._fit!(m, start = [0.0, 0.0, 0.0], keepx = true, keepy = true);

R2 = LSurvResp(int, outt, d);
P2 = PHParms(X);
m2 = PHModel(R2, P2);  #default is "efron" method for ties
@btime res2 = fit!(m2, start = [0.0, 0.0, 0.0], keepx = true, keepy = true);

res
res2

argmax([res.P._LL[end], res2.P._LL[end]])

res.P._LL[end] - res2.P._LL[end]

# in progress functions
# taken from GLM.jl/src/linpred.jl
