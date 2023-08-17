
  struct LSurvCompResp{E<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,W<:AbstractVector, T<:Real, V<:AbstractVector, M<:AbstractMatrix} <: AbstractLSurvResp 
    enter::E
    "`exit`: Time at observation end"
    exit::X
    "`y`: event type in observation (integer)"
    y::Y	
    "`wts`: observation weights"
    wts::W
    "`eventtimes`: unique event times"
    eventtimes::X
    "`origin`: origin on the time scale"
    origin::T
    "`eventtypes`: vector of unique event types"
    eventtypes::V
    "`eventmatrix`: matrix of indicators on the observation level"
    eventmatrix::M
  end
    
  function LSurvCompResp(enter::E, exit::X, y::Y, wts::W) where {E<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,W<:AbstractVector}
    ne  = length(enter)
    nx = length(exit)
    ny = length(y)
    lw = length(wts)
    if !(ne == nx == ny)
        throw(DimensionMismatch("lengths of enter, exit, and y ($ne, $nx, $ny) are not equal"))
    end
    if lw != 0 && lw != ny
        throw(DimensionMismatch("wts must have length $n or length 0 but was $lw"))
    end
    eventtimes = sort(unique(exit[findall(y .> 0)]))
    origin = minimum(enter)
    if lw == 0
      wts = ones(Int64,ny)
    end
    eventtypes = sort(unique(y))
    eventmatrix = reduce(hcat, [y .== e for e in eventtypes[2:end]])
   
    return LSurvCompResp(enter,exit,y,wts,eventtimes,origin,eventtypes,eventmatrix)
  end
  
  
mutable struct KMSurv{G <: LSurvResp} <: AbstractNPSurv  
     R::G        # Survival response
     times::AbstractVector
     surv::Vector{Float64}
     riskset::Vector{Int64}
    end

function KMSurv(R::G) where {G <: LSurvResp}
  times = R.eventtimes
  nt = length(times)
  surv = ones(Float64, nt)
  riskset = zeros(Int64, nt)
  KMSurv(R,times,surv,riskset)
end

"""
   using LSurvival
   using Random
   z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 100);
   enter = zeros(length(t));
   X = hcat(x,z);
   R = LSurvResp(enter, t, Int64.(d), wt)
   mf = KMSurv(R)
   _fit!(mf)

"""
function _fit!(m::KMSurv; 
                weights=nothing, 
                eps = 0.00000001,
                censval=0,
                kwargs...)
   # there is some bad floating point issue with epsilon that should be tracked
   # R handles this gracefully
  # ties allowed
  if !isnothing(weights)
    mf.R.wts = weights
  end
  #_dt = zeros(length(orderedtimes))
  _1mdovern = ones(length(m.times))
  for (_i,tt) in enumerate(m.times)
    R = findall((m.R.exit .>= tt) .& (m.R.enter .< (tt-eps)) ) # risk set index (if in times are very close to other out-times, not using epsilon will make risk sets too big)
    ni = sum(mf.R.wts[R]) # sum of weights in risk set
    di = sum(mf.R.wts[R] .* (m.R.y[R] .> censval) .* (m.R.exit[R] .== tt))
    _1mdovern[_i] = log(1.0 - di/ni)
    m.riskset[_i] = ni
  end
  m.surv = exp.(cumsum(_1mdovern))
  m
end




"""
Kaplan Meier for one observation per unit and no late entry
  (simple function)
"""
function km(t,d; weights=nothing)
  # no ties allowed
  if isnothing(weights) || isnan(weights[1])
    weights = ones(length(t))
  end
  censval = zero(eltype(d))
  orderedtimes = sortperm(t)
  _t = t[orderedtimes]
  _d = d[orderedtimes]
  _weights = weights[orderedtimes]
  whichd = findall( d .> zeros(eltype(d), 1))
  riskset = zeros(Float64, length(t)) # risk set size
 # _dtimes = _t[whichd] # event times
  #_dw = weights[whichd]     # weights at times
  _1mdovern = zeros(Float64, length(_t))
  for (_i,_ti) in enumerate(_t)
    R = findall(_t .>= _ti) # risk set
    ni = sum(_weights[R]) # sum of weights in risk set
    di = _weights[_i]*(_d[_i] .> censval)
    riskset[_i] = ni
    _1mdovern[_i] = 1.0 - di/ni
  end
  _t, cumprod(_1mdovern), riskset
end

"""
Kaplan Meier with late entry, possibly multiple observations per unit
(simple function)
"""
function km(in,out,d; weights=nothing, eps = 0.00000001)
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
  for (_i,tt) in enumerate(orderedtimes)
    R = findall((out .>= tt) .& (in .< (tt-eps)) ) # risk set index (if in times are very close to other out-times, not using epsilon will make risk sets too big)
    ni = sum(weights[R]) # sum of weights in risk set
    di = sum(weights[R] .* (d[R] .> censval) .* (out[R] .== tt))
    _1mdovern[_i] = log(1.0 - di/ni)
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
function aj(in,out,d;dvalues=[1.0, 2.0], weights=nothing, eps = 0.00000001)
  if isnothing(weights) || isnan(weights[1])
    weights = ones(length(in))
  end
  nvals = length(dvalues)
  # overall survival via Kaplan-Meier
  orderedtimes, S, riskset = km(in,out,d, weights=weights, eps=eps) # note ordered times are unique
  Sm1 = vcat(1.0, S)
  ajest = zeros(length(orderedtimes), nvals)
  _d = zeros(length(out), nvals)
  for (jidx,j) in enumerate(dvalues)
    _d[:,jidx] = (d .== j)
  end
  for (_i,tt) in enumerate(orderedtimes)
    R = findall((out .>= tt) .& (in .< (tt-eps))) # risk set
    weightsR = weights[R]
    ni = sum(weightsR) # sum of weights/weighted individuals in risk set
    for (jidx,j) in enumerate(dvalues)
      dij = sum(weightsR .* _d[R,jidx] .* (out[R] .== tt))
      ajest[_i, jidx] = Sm1[_i] * dij/ni
    end
  end
  for jidx in 1:nvals
    ajest[:,jidx] = 1.0 .- cumsum(ajest[:,jidx])
  end
  orderedtimes, S, ajest, riskset
end
;


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
function subdistribution_hazard_cuminc(in,out,d;dvalues=[1.0, 2.0], weights=nothing, eps = 0.0)
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
  @inbounds for (_i,tt) in enumerate(orderedtimes_dmain)
    aliveandatriskidx = findall((in .< (tt-eps)) .&& (out .>= tt))
    hadcompidx = findall((out .<= tt) .&& (d .!= dmain))
    dmain_now = findall((out .== tt) .&& (d .== dmain))
    pseudoR = union(aliveandatriskidx, hadcompidx) 
    casesidx = intersect(dmain_now, aliveandatriskidx)
    ni = sum(weights[pseudoR]) # sum of weights in risk set
    di = sum(weights[casesidx])
    _haz[_i] = di/ni
  end
  orderedtimes_dmain, cumsum(_haz), 1.0 .- exp.(.-cumsum(_haz)), [:times, :cumhaz, :ci]
end
;

"""
Expected number of years of life lost due to cause k

  using Distributions, Plots, Random
  plotly()
  z,x,t,d, event,weights = dgm_comprisk(n=200, rng=MersenneTwister(1232));
  
  times_sd, cumhaz, ci_sd = subdistribution_hazard_cuminc(zeros(length(t)), t, event, dvalues=[1.0, 2.0]);
  times_aj, S, ajest, riskset, events = aalen_johansen(zeros(length(t)), t, event, dvalues=[1.0, 2.0]);
  time0, eyll0 = e_yearsoflifelost(times_aj, 1.0 .- S)  
  time2, eyll1 = e_yearsoflifelost(times_aj, ajest[:,1])  
  time1, eyll2 = e_yearsoflifelost(times_sd, ci_sd)  
  # CI estimates
  plot(times_aj, ajest[:,1], label="AJ", st=:step);
  plot!(times_sd, ci_sd, label="SD", st=:step)  
  # expected years of life lost by time k, given a specific cause or overall
  plot(time0, eyll0, label="Overall", st=:step);
  plot!(time1, eyll1, label="AJ", st=:step);
  plot!(time2, eyll2, label="SD", st=:step)  

"""
function e_yearsoflifelost(time,ci)
  cilag = vcat(0, ci[1:(end-1)])
  difftime = diff(vcat(0,time))
  time, cumsum(cilag .* difftime)  
end


;