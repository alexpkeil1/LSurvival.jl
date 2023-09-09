# to implement

# currently working here:
# Breslow: score, schoenfeld residuals, robust standard error estimate for Cox model
# Efron: score, schoenfeld residuals, robust standard error estimate for Cox model

# currently NOT working here:
# Weighted Efron: score, schoenfeld residuals


using LSurvival, Random, Optim, BenchmarkTools

######################################################################
# residuals from fitted Cox models
######################################################################



"""

Score residuals: Per observation contribution to score function 

```julia
using LSurvival
####################################################################
dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
)
ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow")
r = exp(ft.P._B[1])
# nXp matrix used for schoenfeld and score residuals
# (x(t)-mux(t))*dM(t) 
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
r = exp(ft.P._B[1])

# (x(t)-mux(t))*dM(t)
truthmat = permutedims(hcat(
  [(1-r/(r+1)) * (1-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), (1-r/(r+3)) * (1/2-1/(r+3)) + (1-r/(r+5)) * (1/2-1/(r+5)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (1/2-1/(r+3)) + (0-r/(r+5)) * (1/2-1/(r+5)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-1/(r+3)) +  (0-r/(r+5)) * (0-2/(r+5)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-1/(r+3)) +  (0-r/(r+5)) * (0-2/(r+5)), (0-0) * (1-1)]
))
truth = sum(truthmat, dims=2)[:]
[5/12, -1/12, 55/144, -5/144, 29/144, 29/144] # actual truth under the null
S

S


ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="efron")
r = exp(ft.P._B[1])
S = resid_score(ft)[:]
[0.113278, -0.044234, -0.102920, -0.407840, 0.220858, 0.220858]

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
####################################################################
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
truthmat = [
  (1-r/(r+1)) * (1-r/(3r+3))  0                           0;
  (1-r/(r+1)) * (0-r/(3r+3))  0                           0;
  (1-r/(r+1)) * (0-r/(3r+3))  (1-r/(r+3)) * (1-2r/(r+3))  0;
  (0-r/(r+1)) * (0-1/(3r+3))  (0-r/(r+3)) * (1-2/(r+3))  0;
  (0-r/(r+1)) * (0-1/(3r+3))  (0-r/(r+3)) * (0-2/(r+3))  0;
  (0-r/(r+1)) * (0-1/(3r+3))  (0-r/(r+3)) * (0-2/(r+3))  (0-0) * (1-1);
]
truth = sum(truthmat, dims=1)[:]
all(isapprox.(S, truth))

ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow")

X = ft.P.X
S = resid_schoenfeld(ft)[:]
r = exp(ft.P._B[1])
truthmat = [
  (1-r/(r+1)) * (1-r/(3r+3))  0                           0;
  (1-r/(r+1)) * (0-r/(3r+3))  0                           0;
  (1-r/(r+1)) * (0-r/(3r+3))  (1-r/(r+3)) * (1-2r/(r+3))  0;
  (0-r/(r+1)) * (0-1/(3r+3))  (0-r/(r+3)) * (1-2/(r+3))  0;
  (0-r/(r+1)) * (0-1/(3r+3))  (0-r/(r+3)) * (0-2/(r+3))  0;
  (0-r/(r+1)) * (0-1/(3r+3))  (0-r/(r+3)) * (0-2/(r+3))  (0-0) * (1-1);
]
truth = sum(truthmat, dims=2)[:]
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

"""
using LSurvival
dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
)
ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow")

resid_dfbeta(ft)

"""
function resid_dfbeta(m::M) where {M<:PHModel}
    @warn "Check the sign of these against R"
    L = resid_score(ft)
    H = ft.P._hess
    dfbeta = L * inv(H)
    return dfbeta
end

"""
using LSurvival
dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
)
ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow")

robust_vcov(ft)

"""
function robust_vcov(m::M) where {M<:PHModel}
    dfbeta = resid_dfbeta(ft)
    robVar = dfbeta'dfbeta
    return robVar
end



"""
####################################################################
dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
)

ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow", maxiter=0)

X = ft.P.X
L = resid_Lmat(ft)[1]
r = exp(ft.P._B[1])
truthmat = [
  (1-r/(r+1)) * (1-r/(3r+3))  0                           0;
  (1-r/(r+1)) * (0-r/(3r+3))  0                           0;
  (1-r/(r+1)) * (0-r/(3r+3))  (1-r/(r+3)) * (1-2r/(r+3))  0;
  (0-r/(r+1)) * (0-1/(3r+3))  (0-r/(r+3)) * (1-2/(r+3))  0;
  (0-r/(r+1)) * (0-1/(3r+3))  (0-r/(r+3)) * (0-2/(r+3))  0;
  (0-r/(r+1)) * (0-1/(3r+3))  (0-r/(r+3)) * (0-2/(r+3))  (0-0) * (1-1)
  ]

@assert sum((L .- truthmat).^2)<0.001


ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="efron")

# goal: -0.05868591908545168
#x = 1
#dmt [[-0.14066352629411727, -0.01684851183809455]
# mux = [0.6406635262941173, 0.5168485118380945] # same
#-0.14066352629411727* (x-.6406635262941173) + -0.01684851183809455* (x-.5168485118380945)


L = resid_Lmat(ft)[1]
r = exp(ft.P._B[1])
# (x(t)-mux(t))*dM(t)
truthmat = [
  (1-r/(r+1)) * (1-r/(3r+3))  0                           0;
  (1-r/(r+1)) * (0-r/(3r+3))  0                           0;
#  (1-r/(r+1)) * (0-1/(3r+3))  (1-r/(r+3)) * (1/2-1/(r+3)) + (1-r/(r+5)) * (1/2-1/(r+5))  0; # error in Therneau's calculation
  (1-r/(r+1)) * (0-1/(3r+3))  (1-r/(r+3)) * (1/2-1/(r+3)) + (1-r/(r+5)) * (1/2-1/(r+5))  0;
  (0-r/(r+1)) * (0-1/(3r+3))  (0-r/(r+3)) * (1/2-1/(r+3)) + (0-r/(r+5)) * (1/2-1/(r+5))  0;
  (0-r/(r+1)) * (0-1/(3r+3))  (0-r/(r+3)) * (0-1/(r+3)) +  (0-r/(r+5)) * (0-2/(r+5))  0;
  (0-r/(r+1)) * (0-1/(3r+3))  (0-r/(r+3)) * (0-1/(r+3)) +  (0-r/(r+5)) * (0-2/(r+5))  (0-0) * (1-1);
  ]
sum(truthmat, dims=2)
sum(L, dims=2)

L
truthmat
@assert sum((L .- truthmat).^2)<0.001
####################################################################
dat2 = (
    enter = [1,2,5,2,1,7,3,4,8,8],
    exit = [2,3,6,7,8,9,9,9,14,17],
    status = [1,1,1,1,1,1,1,0,0,0],
    x = [1,0,0,1,0,1,1,1,0,0]
)

ft = coxph(@formula(Surv(enter, exit,status)~x),dat2, keepx=true, keepy=true, ties="breslow", maxiter=0)

L = resid_Lmat(ft)[1]
truthmat = [
    0.25  0.0000000  0.00  0.0000  0.0000  0.00;
    0.00 -0.2222222  0.00  0.0000  0.0000  0.00;
    0.00  0.0000000 -0.48  0.0000  0.0000  0.00;
    0.00 -0.2222222 -0.08  0.1875  0.0000  0.00;
    0.25  0.1111111  0.12  0.1875 -0.5625  0.00;
    0.00  0.0000000  0.00  0.0000 -0.0625  0.24;
    0.00  0.0000000 -0.08 -0.0625 -0.0625  0.24;
    0.00  0.0000000 -0.08 -0.0625 -0.0625 -0.16;
    0.00  0.0000000  0.00  0.0000  0.0000  0.24;
    0.00  0.0000000  0.00  0.0000  0.0000  0.24;
]
@assert all(isapprox.(L, truthmat, atol=0.00001))



ft = coxph(@formula(Surv(enter, exit,status)~x),dat2, keepx=true, keepy=true, ties="breslow")

L = resid_Lmat(ft)[1]
truthmat = [
    0.271565  0.0000000  0.00000000  0.00000000  0.00000000  0.0000000;
    0.000000 -0.2069671  0.00000000  0.00000000  0.00000000  0.0000000;
    0.000000  0.0000000 -0.45771742  0.00000000  0.00000000  0.0000000;
    0.000000 -0.2157089 -0.08122377  0.20107139  0.00000000  0.0000000;
    0.249554  0.1078545  0.12183565  0.19532865 -0.53849041  0.0000000;
    0.000000  0.0000000  0.00000000  0.00000000 -0.06510955  0.2579994;
    0.000000  0.0000000 -0.08122377 -0.06510955 -0.06510955  0.2579994;
    0.000000  0.0000000 -0.08122377 -0.06510955 -0.06510955 -0.1624475;
    0.000000  0.0000000  0.00000000  0.00000000  0.00000000  0.2436713;
    0.000000  0.0000000  0.00000000  0.00000000  0.00000000  0.2436713;
]

@assert all(isapprox.(L, truthmat, atol=0.00001))



ft = coxph(@formula(Surv(enter, exit,status)~x),dat2, keepx=true, keepy=true, ties="efron", maxiter=0)
L = resid_Lmat(ft)[1]

truthmat = [
    0.25  0.0000000  0.00  0.0000  0.0000  0.00  0.0000;
    0.00 -0.2222222  0.00  0.0000  0.0000  0.00  0.0000;
    0.00  0.0000000 -0.48  0.0000  0.0000  0.00  0.0000;
    0.00 -0.2222222 -0.08  0.1875  0.0000  0.00  0.0000;
    0.25  0.1111111  0.12  0.1875 -0.5625  0.00  0.0000;
    0.00  0.0000000  0.00  0.0000 -0.0625  0.12  0.1875;
    0.00  0.0000000 -0.08 -0.0625 -0.0625  0.12  0.1875;
    0.00  0.0000000 -0.08 -0.0625 -0.0625 -0.08 -0.1250;
    0.00  0.0000000  0.00  0.0000  0.0000  0.12  0.1250;
    0.00  0.0000000  0.00  0.0000  0.0000  0.12  0.1250;
]
sum(truthmat, dims=2)
sum(L, dims=2)
@assert all(isapprox.(L, truthmat, atol=0.00001))



######################################################################
dat3 = (
    time = [1,1,2,2,2,2,3,4,5],
    status = [1,0,1,1,1,0,0,1,0],
    x = [2,0,1,1,0,1,0,1,0],
    wt = [1,2,3,4,3,2,1,2,1]
)

ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="breslow", maxiter=0)
r = exp(ft.P._B[1])
dM, dt, di = dexpected_NA(ft);
ft.bh[:,1]
trueλ0 = [1/(r^2 + 11r+7), 10/(11r+5), 2/(2r+1)]
muX = muX_t(ft, di)
truemuX = [(2r^2 + 11)*trueλ0[1],11r/(11r + 5),2r/(2r+1)]

resid_Lmat(ft)[1]
truthmat = [(2 - 13/19)*(1 - 1/19) 0 0;
(0 - 13/19)*(0 - 1/19) 0 0;
(1 - 13/19)*(0 - 1/19)  (1 - 11/16)*(1 - 5/8) 0;
(1 - 13/19)*(0 - 1/19)  (1 - 11/16)*(1 - 5/8) 0;
(0 - 13/19)*(0 - 1/19)  (0 - 11/16)*(1 - 5/8) 0;
(1 - 13/19)*(0 - 1/19)  (1 - 11/16)*(0 - 5/8) 0;
(0 - 13/19)*(0 - 1/19)  (0 - 11/16)*(0 - 5/8) 0;
(1 - 13/19)*(0 - 1/19)  (1 - 11/16)*(0 - 5/8) (1 - 2/3)*(1 - 2/3);
(0 - 13/19)*(0 - 1/19)  (0 - 11/16)*(0 - 5/8) (0 - 2/3)*(0 - 2/3)]



resid_score(ft)
truescore = sum(truthmat, dims=2)


ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="breslow")
resid_score(ft)


ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="efron", maxiter=0)
resid_score(ft)
L = resid_Lmat(ft)[1]

ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="efron")
resid_score(ft)
L = resid_Lmat(ft)[1]


"""
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
    wts = m.R.wts
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
    maxties = maximum(ties)
    #

    dMt, dt, di = dexpected_FH(m)
    muXt = muX_tE(m, di)

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

# goal: -0.452075
#x = 0
#dmt [0.3802211754313724, 0.40336970236761893]
# mux = [0.6406635262941173, 0.5168485118380945] # same
#.3802211754313724* (x-.6406635262941173) + 0.40336970236761893* (x-.5168485118380945)




"""
dat3 = (
    time = [1,1,2,2,2,2,3,4,5],
    status = [1,0,1,1,1,0,0,1,0],
    x = [2,0,1,1,0,1,0,1,0],
    wt = [1,2,3,4,3,2,1,2,1]
)

ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="breslow", maxiter=0)
dM, dt, di = dexpected_NA(ft);
muX = muX_t(ft, di)
truemuX = [13/19,11/16,2/3]

"""
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
    # which event times is each observation at risk?
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
                if (y[i] > 0) && (exit[i] == bht[t])
                    ew = LSurvival.efron_weights(ties[whichbhindex[i][t]])
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

"""
using LSurvival
dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
)
ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="efron", maxiter=0)

r = exp(ft.P._B[1])
truthmat = permutedims(hcat(
  [(1-r/(r+1)) * (1-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), (1-r/(r+3)) * (1/2-1/(r+3)) + (1-r/(r+5)) * (1/2-1/(r+5)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (1/2-1/(r+3)) + (0-r/(r+5)) * (1/2-1/(r+5)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-1/(r+3)) +  (0-r/(r+5)) * (0-2/(r+5)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-1/(r+3)) +  (0-r/(r+5)) * (0-2/(r+5)), (0-0) * (1-1)]
))


LSurvival.expected_FH(ft)
dE, dt, di = dexpected_FH(ft)

"""
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
    #dE = [ dER[whichbhindex[i]] for i in eachindex(whichbhindex)]
    for i in eachindex(whichbhindex)
        if length(whichbhcaseindex[i]) > 0
            xix = findall(whichbhindex[i] .== whichbhcaseindex[i])
            dE[i][xix] = rr[i] .* dE0[whichbhcaseindex[i]]
            #dE[i][xix] = dE0[whichbhcaseindex[i]]
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

if false
    ######################################################################
    # fitting with optim (works, but more intensive than defaults)
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
            fc = findall(
                (m.R.y .> 0) .&& isapprox.(m.R.exit, _outj) .&& (m.R.enter .< _outj),
            )
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

end # if false


if false
    id, int, outt, data =
        LSurvival.dgm(MersenneTwister(345), 100, 10; afun = LSurvival.int_0)
    data[:, 1] = round.(data[:, 1], digits = 3)
    d, X = data[:, 4], data[:, 1:3]


    # not-yet-fit PH model object
    #m = PHModel(R, P, "breslow")
    #LSurvival._fit!(m, start = [0.0, 0.0, 0.0], keepx=true, keepy=true)
    #isfitted(m)
    R = LSurvResp(int, outt, d)
    P = PHParms(X)
    m = PHModel(R, P)  #default is "efron" method for ties
    @btime res = LSurvival._fit!(m, start = [0.0, 0.0, 0.0], keepx = true, keepy = true)

    R2 = LSurvResp(int, outt, d)
    P2 = PHParms(X)
    m2 = PHModel(R2, P2)  #default is "efron" method for ties
    @btime res2 = fit!(m2, start = [0.0, 0.0, 0.0], keepx = true, keepy = true)

    res
    res2

    argmax([res.P._LL[end], res2.P._LL[end]])

    res.P._LL[end] - res2.P._LL[end]

    # in progress functions
    # taken from GLM.jl/src/linpred.jl
end
