
function _coxrisk!(_r, X, B)
    map!(z -> exp(z), _r, X * B)
    nothing
end

function _coxrisk(X, B)
    _r = ones(size(X, 1))
    _coxrisk!(_r, X, B)
end

"""
deprecated function
"""
function lgh!(lowermethod3, _den, _LL, _grad, _hess, j, p, X, _r, _wt, caseidx, risksetidx)
    whichmeth = findfirst(lowermethod3 .== ["efr", "bre"])
    isnothing(whichmeth) ? throw("Ties method not recognized") : true
    if whichmeth == 1
        lgh_efron!(
            _den,
            _LL,
            _grad,
            _hess,
            j,
            p,
            X[caseidx, :],
            X[risksetidx, :],
            _r[caseidx],
            _r[risksetidx],
            _wt[caseidx],
            _wt[risksetidx],
            length(caseidx),  # nties
        )
    elseif whichmeth == 2
        lgh_breslow!(
            _den,
            _LL,
            _grad,
            _hess,
            j,
            p,
            X[caseidx, :],
            X[risksetidx, :],
            _r[caseidx],
            _r[risksetidx],
            _wt[caseidx],
            _wt[risksetidx],
        )
    end
end

"""
Deprecated function
"""
function lgh_breslow!(
    _den,
    _LL,
    _grad,
    _hess,
    j,
    p,
    Xcases,
    Xriskset,
    _rcases,
    _rriskset,
    _wtcases,
    _wtriskset,
)
    den = sum(_rriskset .* _wtriskset)
    _LL .+= sum(_wtcases .* log.(_rcases)) .- log(den) * sum(_wtcases)
    #
    numg = Xriskset' * (_rriskset .* _wtriskset)
    xbar = numg / den # risk-score-weighted average of X columns among risk set
    _grad .+= (Xcases .- xbar')' * (_wtcases)
    #
    numgg = (Xriskset' * Diagonal(_rriskset .* _wtriskset) * Xriskset)
    xxbar = numgg / den
    _hess .+= -(xxbar - xbar * xbar') * sum(_wtcases)
    _den[j] = den
    nothing
end

"""
Deprecated function
"""
function lgh_efron!(
    _den,
    _LL,
    _grad,
    _hess,
    j,
    p,
    Xcases,
    Xriskset,
    _rcases,
    _rriskset,
    _wtcases,
    _wtriskset,
    nties,
)

    effwts = efron_weights(nties)
    den = sum(_wtriskset .* _rriskset)
    denc = sum(_wtcases .* _rcases)
    dens = [den - denc * ew for ew in effwts]
    _LL .+= sum(_wtcases .* log.(_rcases)) .- sum(log.(dens)) * 1 / nties * sum(_wtcases) # gives same answer as R with weights
    #
    numg = Xriskset' * (_wtriskset .* _rriskset)
    numgs = [numg .- ew * Xcases' * (_wtcases .* _rcases) for ew in effwts]
    xbars = numgs ./ dens # risk-score-weighted average of X columns among risk set
    _grad .+= Xcases' * _wtcases
    for i = 1:nties
        _grad .+= (-xbars[i]) * sum(_wtcases) / nties
    end
    numgg = (Xriskset' * Diagonal(_wtriskset .* _rriskset) * Xriskset)
    numggs =
        [numgg .- ew .* Xcases' * Diagonal(_wtcases .* _rcases) * Xcases for ew in effwts]
    xxbars = numggs ./ dens
    #
    for i = 1:nties
        _hess .-= (xxbars[i] - xbars[i] * xbars[i]') .* sum(_wtcases) / nties
    end
    #_den[j] = den # Breslow estimator
    sw = sum(_wtcases)
    aw = sw / nties
    _den[j] = 1.0 ./ sum(aw ./ dens) # using Efron estimator
    nothing
end

function _stepcox!(
    lowermethod3,
    # recycled parameters
    _LL::Vector,
    _grad::Vector,
    _hess::Matrix{Float64},
    # data
    _in::Vector,
    _out::Vector,
    d::Union{Vector,BitVector},
    X,
    _wt::Vector,
    # fixed parameters
    _B::Vector,
    # indexs
    p::T,
    n::U,
    eventtimes::Vector,
    # recycled containers
    _r::Vector,
    # big indexes
    risksetidxs,
    caseidxs,
) where {T<:Int,U<:Int}
    _coxrisk!(_r, X, _B) # updates all elements of _r as exp(X*_B)
    # loop over event times
    ne = length(eventtimes)
    den, wtdriskset, wtdcases = zeros(ne), zeros(ne), zeros(ne)
    _LL .*= 0.0
    _grad .*= 0.0
    _hess .*= 0.0
    @inbounds for j = 1:ne
        risksetidx = risksetidxs[j]
        caseidx = caseidxs[j]
        lgh!(lowermethod3, den, _LL, _grad, _hess, j, p, X, _r, _wt, caseidx, risksetidx)
        wtdriskset[j] = sum(_wt[risksetidx])
        wtdcases[j] = sum(_wt[caseidx])
    end # j
    den, wtdriskset, wtdcases
end #function _stepcox!

##################################################################################################################### 
# Newton raphson wrapper function
############################################################################################################d########

"""
Deprecated function

Estimate parameters of an extended Cox model

Using: Newton raphson algorithm with modified/adaptive step sizes

Keyword inputs:
method="efron", 
inits=nothing , # initial parameter values, set to zero if this is nothing
tol=10e-9,      #  convergence tolerance based on log likelihod: likrat = abs(lastLL/_LL[1]), absdiff = abs(lastLL-_LL[1]), reldiff = max(likrat, inv(likrat)) -1.0
maxiter=500    # maximum number of iterations for Newton Raphson algorithm (set to zero to calculate likelihood, gradient, Hessian at the initial parameter values)

Outputs:
beta: coefficients 
ll: log partial likelihood history (all iterations), with final value being the (log) maximum partial likelihood (log-MPL)
g: gradient vector (first derivative of log partial likelihood) at log-MPL
h: hessian matrix (second derivative of log partial likelihood) at log-MPL
basehaz: Matrix: baseline hazard at referent level of all covariates, weighted risk set size, weighted # of cases, time


Examples: 
```julia-repl   
  using LSurvival
  # simulating discrete survival data for 20 individuals at 10 time points
  id, int, outt, data = LSurvival.dgm(20, 5;afun=LSurvival.int_0);
  
  d,X = data[:,4], data[:,1:3]
  
  # getting raw values of output
  args = (int, outt, d, X)
  beta, ll, g, h, basehaz = coxmodel(args..., method="efron")
  beta2, ll2, g2, h2, basehaz2 = coxmodel(args..., method="breslow")


  # nicer summary of results
  args = (int, outt, d, X)
  res = coxmodel(args..., method="efron");
  coxsum = cox_summary(res, alpha=0.05, verbose=true);
    
```
"""
function coxmodel(
    _in::Array{<:Real,1},
    _out::Array{<:Real,1},
    d::Array{<:Real,1},
    X::Array{<:Real,2};
    weights = nothing,
    method = "efron",
    inits = nothing,
    tol = 10e-9,
    maxiter = 500,
)
    #(_in::Array{Float64}, _out::Array{Float64}, d, X::Array{Float64,2}, _wt::Array{Float64})=args
    #### move #####
    if isnothing(weights)
        weights = ones(size(_in, 1))
    end
    if size(_out, 1) == 0
        throw("error in function call")
    end
    conts = containers(_in, _out, d, X, weights, inits)
    #(n,p,eventidx, eventtimes,nevents,_B,_r, _basehaz, _riskset,_LL,_grad,_hess) = conts
    (n, p, eventtimes, _B, _r, _LL, _grad, _hess) = conts
    #
    lowermethod3 = lowercase(method[1:3])
    # tuning params
    totiter = 0
    λ = 1.0
    absdiff = tol * 2.0
    oldQ = floatmax()
    #bestb = _B
    lastLL = -floatmax()
    risksetidxs, caseidxs = [], []
    @inbounds for _outj in eventtimes
        push!(risksetidxs, findall((_in .< _outj) .&& (_out .>= _outj)))
        push!(caseidxs, findall((d .> 0) .&& isapprox.(_out, _outj) .&& (_in .< _outj)))
    end
    den, _sumwtriskset, _sumwtcase = _stepcox!(
        lowermethod3,
        _LL,
        _grad,
        _hess,
        _in,
        _out,
        d,
        X,
        weights,
        _B,
        p,
        n,
        eventtimes,
        _r,
        risksetidxs,
        caseidxs,
    )
    _llhistory = [_LL[1]] # if inits are zero, 2*(_llhistory[end] - _llhistory[1]) is the likelihood ratio test on all predictors
    converged = false
    # repeat newton raphson steps until convergence or max iterations
    while totiter < maxiter
        totiter += 1
        ######
        # update 
        #######
        Q = 0.5 * (_grad' * _grad) #modified step size if gradient increases
        likrat = (lastLL / _LL[1])
        absdiff = abs(lastLL - _LL[1])
        reldiff = max(likrat, inv(likrat)) - 1.0
        converged = (reldiff < tol) || (absdiff < sqrt(tol))
        if converged
            break
        end
        if Q > oldQ
            λ *= 0.5  # step-halving
        else
            λ = min(2.0λ, 1.0) # de-halving
            #bestb = _B
            nothing
        end
        isnan(_LL[1]) ? throw("Log-partial-likelihood is NaN") : true
        if abs(_LL[1]) != Inf
            _B .+= inv(-(_hess)) * _grad .* λ # newton raphson step
            oldQ = Q
        else
            throw("log-partial-likelihood is infinite")
        end
        lastLL = _LL[1]
        den, _, _ = _stepcox!(
            lowermethod3,
            _LL,
            _grad,
            _hess,
            _in,
            _out,
            d,
            X,
            weights,
            _B,
            p,
            n,
            eventtimes,
            _r,
            risksetidxs,
            caseidxs,
        )
        push!(_llhistory, _LL[1])
    end
    if totiter == maxiter
        @warn "Algorithm did not converge after $totiter iterations"
    end
    if lowermethod3 == "bre"
        bh = [_sumwtcase ./ den _sumwtriskset _sumwtcase eventtimes]
    elseif lowermethod3 == "efr"
        bh = [1.0 ./ den _sumwtriskset _sumwtcase eventtimes]
    end
    (_B, _llhistory, _grad, _hess, bh)
end;
coxmodel(_out::Array{<:Real,1}, d::Array{<:Real,1}, X::Array{<:Real,2}; kwargs...) =
    coxmodel(zeros(typeof(_out), length(_out)), _out, d, X; kwargs...)



function cox_summary(args; alpha = 0.05, verbose = true)
    beta, ll, g, h, basehaz = args
    std_err = sqrt.(diag(-inv(h)))
    z = beta ./ std_err
    zcrit = quantile.(Distributions.Normal(), [alpha / 2.0, 1.0 - alpha / 2.0])
    lci = beta .+ zcrit[1] * std_err
    uci = beta .+ zcrit[2] * std_err
    pval = calcp.(z)
    op = hcat(beta, std_err, lci, uci, z, pval)
    verbose ? true : return (op)
    chi2 = ll[end] - ll[1]
    df = length(beta)
    lrtp = 1 - cdf(Distributions.Chisq(df), chi2)
    head = ["ln(HR)", "StdErr", "LCI", "UCI", "Z", "P(>|Z|)"]
    rown = ["b$i" for i = 1:size(op)[1]]
    coeftab = CoefTable(op, head, rown, 6, 5)
    iob = IOBuffer()
    println(iob, coeftab)
    str = """\nMaximum partial likelihood estimates (alpha=$alpha):\n"""
    str *= String(take!(iob))
    str *= "Partial log-likelihood (null): $(@sprintf("%8g", ll[1]))\n"
    str *= "Partial log-likelihood (fitted): $(@sprintf("%8g", ll[end]))\n"
    str *= "LRT p-value (X^2=$(round(chi2, digits=2)), df=$df): $(@sprintf("%5g", lrtp))\n"
    str *= "Newton-Raphson iterations: $(length(ll)-1)"
    println(str)
    op
end


"""
Deprecated function
  Estimating cumulative incidence from two or more cause-specific Cox models
  
  z,x,outt,d,event,weights = LSurvival.dgm_comprisk(120)
  X = hcat(z,x)
  int = zeros(120)
  d1  = d .* Int.(event.== 1)
  d2  = d .* Int.(event.== 2)
  sum(d)/length(d)
  
  
  lnhr1, ll1, g1, h1, bh1 = coxmodel(int, outt, d1, X, method="efron");
  lnhr2, ll2, g2, h2, bh2 = coxmodel(int, outt, d2, X, method="efron");
  bhlist = [bh1, bh2]
  coeflist = [lnhr1, lnhr2]
  covarmat = sum(X, dims=1) ./ size(X,1)
  ci, surv = ci_from_coxmodels(bhlist;eventtypes=[1,2], coeflist=coeflist, covarmat=covarmat)
  ci, surv = ci_from_coxmodels(bhlist;eventtypes=[1,2])
  """
function ci_from_coxmodels(
    bhlist;
    eventtypes = [1, 2],
    coeflist = nothing,
    covarmat = nothing,
)
    bhlist = [hcat(bh, fill(eventtypes[i], size(bh, 1))) for (i, bh) in enumerate(bhlist)]
    bh = reduce(vcat, bhlist)
    sp = sortperm(bh[:, 4])
    bh = bh[sp, :]
    ntimes::Int = size(bh, 1)
    ci, surv, hr = zeros(Float64, ntimes, length(eventtypes)),
    fill(1.0, ntimes),
    ones(Float64, length(eventtypes))
    ch::Float64 = 0.0
    lsurv::Float64 = 1.0
    if !isnothing(coeflist)
        @inbounds for (j, d) in enumerate(eventtypes)
            hr[j] = exp(dot(covarmat, coeflist[j]))
        end
    end
    lci = zeros(length(eventtypes))
    @inbounds for i = 1:ntimes
        @inbounds for (j, d) in enumerate(eventtypes)
            if bh[i, 5] == d
                bh[i, 1] *= hr[j]
                ci[i, j] = lci[j] + bh[i, 1] * lsurv
            else
                ci[i, j] = lci[j]
            end
        end
        ch += bh[i, 1]
        surv[i] = exp(-ch)
        lsurv = surv[i]
        lci = ci[i, :]
    end
    ci, surv, bh[:, 5], bh[:, 4]
end


function containers(in, out, d, X, wt, inits)
    if size(out, 1) == 0
        throw("error in function call")
    end
    if isnothing(wt)
        wt = ones(size(in, 1))
    end
    @assert length(size(X)) == 2
    n, p = size(X)
    # indexes,counters
    #eventidx = findall(d .> 0)
    eventtimes = sort(unique(out[findall(d .> 0)]))
    #nevents = length(eventidx);
    # containers
    _B = isnothing(inits) ? zeros(p) : copy(inits)
    _r = zeros(Float64, n)
    #_basehaz = zeros(Float64, nevents) # baseline hazard estimate
    #_riskset = zeros(Int, nevents) # baseline hazard estimate
    _LL = zeros(1)
    _grad = zeros(p)
    _hess = zeros(p, p) #initialize
    #(n,p,eventidx, eventtimes,nevents,_B,_r, _basehaz, _riskset,_LL,_grad,_hess)
    n, p, eventtimes, _B, _r, _LL, _grad, _hess
end




"""
Kaplan Meier for one observation per unit and no late entry
  (simple function)
"""
function km(t, d; weights = nothing)
    # no ties allowed
    if isnothing(weights) || isnan(weights[1])
        weights = ones(length(t))
    end
    censval = zero(eltype(d))
    orderedtimes = sortperm(t)
    _t = t[orderedtimes]
    _d = d[orderedtimes]
    _weights = weights[orderedtimes]
    whichd = findall(d .> zeros(eltype(d), 1))
    riskset = zeros(Float64, length(t)) # risk set size
    # _dtimes = _t[whichd] # event times
    #_dw = weights[whichd]     # weights at times
    _1mdovern = zeros(Float64, length(_t))
    for (_i, _ti) in enumerate(_t)
        R = findall(_t .>= _ti) # risk set
        ni = sum(_weights[R]) # sum of weights in risk set
        di = _weights[_i] * (_d[_i] .> censval)
        riskset[_i] = ni
        _1mdovern[_i] = 1.0 - di / ni
    end
    _t, cumprod(_1mdovern), riskset
end

"""
Kaplan Meier with late entry, possibly multiple observations per unit
(simple function)
"""
function km(in, out, d; weights = nothing, eps = 0.00000001)
    # there is some bad floating point issue with epsilon that should be tracked
    # R handles this gracefully
    # ties allowed
    if isnothing(weights) || isnan(weights[1])
        weights = ones(length(in))
    end
    censval = zero(eltype(d))
    times = unique(out)
    orderedtimes = sort(times)
    riskset = zeros(Float64, length(times)) # risk set size
    #_dt = zeros(length(orderedtimes))
    _1mdovern = ones(length(orderedtimes))
    for (_i, tt) in enumerate(orderedtimes)
        R = findall((out .>= tt) .& (in .< (tt - eps))) # risk set index (if in times are very close to other out-times, not using epsilon will make risk sets too big)
        ni = sum(weights[R]) # sum of weights in risk set
        di = sum(weights[R] .* (d[R] .> censval) .* (out[R] .== tt))
        _1mdovern[_i] = log(1.0 - di / ni)
        riskset[_i] = ni
    end
    orderedtimes, exp.(cumsum(_1mdovern)), riskset
end

# i = 123
# tt = orderedtimes[_i]

"""
Aalen-Johansen (survival) with late entry, possibly multiple observations per unit
  (simple function)
"""
function aj(in, out, d; dvalues = [1.0, 2.0], weights = nothing, eps = 0.00000001)
    if isnothing(weights) || isnan(weights[1])
        weights = ones(length(in))
    end
    nvals = length(dvalues)
    # overall survival via Kaplan-Meier
    orderedtimes, S, riskset = km(in, out, d, weights = weights, eps = eps) # note ordered times are unique
    Sm1 = vcat(1.0, S)
    ajest = zeros(length(orderedtimes), nvals)
    _d = zeros(length(out), nvals)
    for (jidx, j) in enumerate(dvalues)
        _d[:, jidx] = (d .== j)
    end
    for (_i, tt) in enumerate(orderedtimes)
        R = findall((out .>= tt) .& (in .< (tt - eps))) # risk set
        weightsR = weights[R]
        ni = sum(weightsR) # sum of weights/weighted individuals in risk set
        for (jidx, j) in enumerate(dvalues)
            dij = sum(weightsR .* _d[R, jidx] .* (out[R] .== tt))
            ajest[_i, jidx] = Sm1[_i] * dij / ni
        end
    end
    for jidx = 1:nvals
        ajest[:, jidx] = 1.0 .- cumsum(ajest[:, jidx])
    end
    orderedtimes, S, ajest, riskset
end;


"""
Kaplan Meier with late entry, possibly multiple observations per unit

Usage: kaplan_meier(in,out,d; weights=nothing, eps = 0.00000001)

  - in = time at entry (numeric vector)
  - out = time at exit (numeric vector)
  - d = event indicator (numeric or boolean vector)

  keywords:
  - weights = vector of observation weights, or nothing (default)
  - eps = (default = 0.00000001) very small numeric value that helps in case of tied times that become misordered due to floating point errors
  
  Output: tuple with entries
  - times: unique event times
  - survival: product limit estimator of survival 
  - riskset: number of uncensored observations used in calculating survival at each event time
  - names = vector of symbols [:times, :surv_overall, :riskset] used as a mnemonic for the function output

"""
#=
function kaplan_meier(in,out,d; weights=nothing, eps = 0.0)
   # there is some bad floating point issue with epsilon that should be tracked
   # R handles this gracefully
  # ties allowed
  if isnothing(weights) || isnan(weights[1])
    weights = ones(length(in))
  end
  censval = zero(eltype(d))
  times = unique(out)
  orderedtimes = sort(times)
  riskset = zeros(Float64, length(times)) # risk set size
  #_dt = zeros(length(orderedtimes))
  _1mdovern = ones(length(orderedtimes))
  @inbounds for (_i,tt) in enumerate(orderedtimes)
    R = findall((out .>= tt) .& (in .< (tt-eps)) ) # risk set index (if in times are very close to other out-times, not using epsilon will make risk sets too big)
    ni = sum(weights[R]) # sum of weights in risk set
    di = sum(weights[R] .* (d[R] .> censval) .* (out[R] .== tt))
    _1mdovern[_i] = log(1.0 - di/ni)
    riskset[_i] = ni
  end
  orderedtimes, exp.(cumsum(_1mdovern)), riskset, [:times, :surv_overall, :riskset]
end
=#

"""
Aalen-Johansen (cumulative incidence) with late entry, possibly multiple observations per unit, non-repeatable events
Usage: aalen_johansen(in,out,d;dvalues=[1.0, 2.0], weights=nothing, eps = 0.00000001)

  - in = time at entry (numeric vector)
  - out = time at exit (numeric vector)
  - d = event indicator (numeric or boolean vector)

  keywords:
  - dvalues = (default = [1.0, 2.0]) a vector of the unique values of 'd' that indicate event types. By default, d is expected to take on values 0.0,1.0,2.0 for 3 event types (censored, event type 1, event type 2)
  - weights = vector of observation weights, or nothing (default)
  - eps = (default = 0.00000001) very small numeric value that helps in case of tied times that become misordered due to floating point errors
  
  Output: tuple with entries
    - times: unique event times
    - survival: product limit estimator of overall survival (e.g. cumulative probability that d is 0.0)
    - ci: Aalen-Johansen estimators of cumulative incidence for each event type. 1-sum of the CI for all event types is equal to overall survival.
    - riskset: number of uncensored observations used in calculating survival at each event time
    - events: number of events of each type used in calculating survival and cumulative incidence at each event time
    - names: vector of symbols [:times, :surv_km_overall, :ci_aalenjohansen, :riskset, :events] used as a mnemonic for the function output

"""
#=
function aalen_johansen(in,out,d;dvalues=[1.0, 2.0], weights=nothing, eps = 0.0)
  if isnothing(weights) || isnan(weights[1])
    weights = ones(length(in))
  end
  nvals = length(dvalues)
  # overall survival via Kaplan-Meier
  orderedtimes, S, riskset, _ = kaplan_meier(in,out,d, weights=weights, eps=eps) # note ordered times are unique
  Sm1 = vcat(1.0, S)
  ajest = zeros(length(orderedtimes), nvals)
  _dij = zeros(length(orderedtimes), nvals)
  _d = zeros(length(out), nvals)
  @inbounds for (jidx,j) in enumerate(dvalues)
    _d[:,jidx] = (d .== j)
  end
  @inbounds for (_i,tt) in enumerate(orderedtimes)
    R = findall((out .>= tt) .& (in .< (tt-eps))) # risk set
    weightsR = weights[R]
    ni = sum(weightsR) # sum of weights/weighted individuals in risk set
    @inbounds for (jidx,j) in enumerate(dvalues)
        # = dij
        _dij[_i,jidx] = sum(weightsR .* _d[R,jidx] .* (out[R] .== tt))
        ajest[_i, jidx] = Sm1[_i] * _dij[_i,jidx]/ni
    end
  end
  for jidx in 1:nvals
    ajest[:,jidx] = cumsum(ajest[:,jidx])
  end
  orderedtimes, S, ajest, riskset, _dij, [:times, :surv_km_overall, :ci_aalenjohansen, :riskset, :events]
end
;
=#

"""
 Non-parametric sub-distribution hazard estimator
  estimating cumulative incidence via the subdistribution hazard function

Usage: subdistribution_hazard_cuminc(in,out,d;dvalues=[1.0, 2.0], weights=nothing, eps = 0.00000001)

  - in = time at entry (numeric vector)
  - out = time at exit (numeric vector)
  - d = event indicator (numeric or boolean vector)
  
  keywords:
  - dvalues = (default = [1.0, 2.0]) a vector of the unique values of 'd' that indicate event types. By default, d is expected to take on values 0.0,1.0,2.0 for 3 event types (censored, event type 1, event type 2)
  - weights = vector of observation weights, or nothing (default)
  - eps = (default = 0.00000001) very small numeric value that helps in case of tied times that become misordered due to floating point errors
  
  Output: tuple with entries
   - times: unique event times
   - cumhaz: cumulative subdistrution hazard for each event type
   - ci: Subdistrution hazard estimators of cumulative incidence for each event type. 1-sum of the CI for all event types is equal to overall survival.
   - events: number of events of each type used in calculating survival and cumulative incidence at each event time
   - names: vector of symbols [:times, :cumhaz, :ci] used as a mnemonic for the function output

Note: 
  For time specific subdistribution hazard given by 'sdhaz(t)', the cumulative incidence for a specific event type calculated over time is 
  
  1.0 .- exp.(.-cumsum(sdhaz(t)))

Examples: 
```julia-repl   
  using LSurvival, Random

  z,x,t,d, event,weights = LSurvival.dgm_comprisk(1000);
  
  # compare these two approaches, where Aalen-Johansen method requires having cause specific hazards for every event type
  times_sd, cumhaz, ci_sd = subdistribution_hazard_cuminc(zeros(length(t)), t, event, dvalues=[1.0, 2.0]);
  times_aj, surv, ajest, riskset, events = aalen_johansen(zeros(length(t)), t, event, dvalues=[1.0, 2.0]);
  
```
"""
function subdistribution_hazard_cuminc(
    in,
    out,
    d;
    dvalues = [1.0, 2.0],
    weights = nothing,
    eps = 0.0,
)
    @warn "This function is not appropriate for data with censoring"
    # ties allowed
    dmain = dvalues[1]
    if isnothing(weights) || isnan(weights[1])
        weights = ones(length(in))
    end
    censval = zero(eltype(d))
    times_dmain = unique(out[findall(d .== dmain)])
    orderedtimes_dmain = sort(times_dmain)
    _haz = ones(length(orderedtimes_dmain))
    @inbounds for (_i, tt) in enumerate(orderedtimes_dmain)
        aliveandatriskidx = findall((in .< (tt - eps)) .&& (out .>= tt))
        hadcompidx = findall((out .<= tt) .&& (d .!= dmain))
        dmain_now = findall((out .== tt) .&& (d .== dmain))
        pseudoR = union(aliveandatriskidx, hadcompidx)
        casesidx = intersect(dmain_now, aliveandatriskidx)
        ni = sum(weights[pseudoR]) # sum of weights in risk set
        di = sum(weights[casesidx])
        _haz[_i] = di / ni
    end
    orderedtimes_dmain, cumsum(_haz), 1.0 .- exp.(.-cumsum(_haz)), [:times, :cumhaz, :ci]
end;


"""
$DOC_E_YEARSOFLIFELOST
"""
function e_yearsoflifelost(time, ci)
    cilag = vcat(0, ci[1:(end-1)])
    difftime = diff(vcat(0, time))
    time, cumsum(cilag .* difftime)
end
