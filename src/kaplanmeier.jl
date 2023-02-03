
"""
Generating data with competing risks
"""
function dgm_comprisk(;n=100, rng=MersenneTwister())
  z = rand(rng, n) .*5
  x = rand(rng, n) .*5
  dt1 = Weibull.(fill(0.75, n), inv.(exp.(-x .- z)))
  dt2 = Weibull.(fill(0.75, n), inv.(exp.(-x .- z)))
  t01 = rand.(rng, dt1)
  t02 = rand.(rng, dt2)
  t0 = min.(t01, t02)
  t = Array{Float64,1}(undef, n)
  for i in 1:n
    t[i] = t0[i] > 1.0 ? 1.0 : t0[i]
  end
  d = (t .== t0)
  event = (t .== t01) .+ 2.0.*(t .== t02)
  wtu = rand(rng, n) .* 5.0
  wt = wtu ./ mean(wtu)
  reshape(round.(z, digits=4), (n,1)), reshape(round.(x, digits=4), (n,1)) ,round.(t, digits=4),d, event, round.(wt, digits=4)
end



"""
Kaplan Meier for one observation per unit and no late entry
"""
function km(t,d; wt=nothing)
  # no ties allowed
  if isnothing(wt) || isnan(wt[1])
    wt = ones(length(t))
  end
  censval = zero(eltype(d))
  orderedtimes = sortperm(t)
  _t = t[orderedtimes]
  _d = d[orderedtimes]
  _wt = wt[orderedtimes]
  whichd = findall( d .> zeros(eltype(d), 1))
  riskset = zeros(Float64, length(t)) # risk set size
 # _dtimes = _t[whichd] # event times
  #_dw = wt[whichd]     # weights at times
  _1mdovern = zeros(Float64, length(_t))
  for (_i,_ti) in enumerate(_t)
    R = findall(_t .>= _ti) # risk set
    ni = sum(_wt[R]) # sum of weights in risk set
    di = _wt[_i]*(_d[_i] .> censval)
    riskset[_i] = ni
    _1mdovern[_i] = 1.0 - di/ni
  end
  _t, cumprod(_1mdovern), riskset
end

"""
Kaplan Meier with late entry, possibly multiple observations per unit
"""
function km(in,out,d; wt=nothing, eps = 0.00000001)
   # there is some bad floating point issue with epsilon that should be tracked
   # R handles this gracefully
  # ties allowed
  if isnothing(wt) || isnan(wt[1])
    wt = ones(length(in))
  end
  censval = zero(eltype(d))
  times = unique(out)
  orderedtimes = sort(times)
  riskset = zeros(Float64, length(times)) # risk set size
  #_dt = zeros(length(orderedtimes))
  _1mdovern = ones(length(orderedtimes))
  for (_i,tt) in enumerate(orderedtimes)
    R = findall((out .>= tt) .& (in .< (tt-eps)) ) # risk set index (if in times are very close to other out-times, not using epsilon will make risk sets too big)
    ni = sum(wt[R]) # sum of weights in risk set
    di = sum(wt[R] .* (d[R] .> censval) .* (out[R] .== tt))
    _1mdovern[_i] = log(1.0 - di/ni)
    riskset[_i] = ni
  end
  orderedtimes, exp.(cumsum(_1mdovern)), riskset
end

# i = 123
# tt = orderedtimes[_i]

"""
Aalen-Johansen (survival) with late entry, possibly multiple observations per unit
"""
function aj(in,out,d;dvalues=[1.0, 2.0], wt=nothing, eps = 0.00000001)
  if isnothing(wt) || isnan(wt[1])
    wt = ones(length(in))
  end
  nvals = length(dvalues)
  # overall survival via Kaplan-Meier
  orderedtimes, S, riskset = km(in,out,d, wt=wt, eps=eps) # note ordered times are unique
  Sm1 = vcat(1.0, S)
  ajest = zeros(length(orderedtimes), nvals)
  _d = zeros(length(out), nvals)
  for (jidx,j) in enumerate(dvalues)
    _d[:,jidx] = (d .== j)
  end
  for (_i,tt) in enumerate(orderedtimes)
    R = findall((out .>= tt) .& (in .< (tt-eps))) # risk set
    wtR = wt[R]
    ni = sum(wtR) # sum of weights/weighted individuals in risk set
    for (jidx,j) in enumerate(dvalues)
      dij = sum(wtR .* _d[R,jidx] .* (out[R] .== tt))
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
just a differently named function with more explicit output
"""
function kaplan_meier(in,out,d; wt=nothing, eps = 0.00000001)
   # there is some bad floating point issue with epsilon that should be tracked
   # R handles this gracefully
  # ties allowed
  if isnothing(wt) || isnan(wt[1])
    wt = ones(length(in))
  end
  censval = zero(eltype(d))
  times = unique(out)
  orderedtimes = sort(times)
  riskset = zeros(Float64, length(times)) # risk set size
  #_dt = zeros(length(orderedtimes))
  _1mdovern = ones(length(orderedtimes))
  @inbounds for (_i,tt) in enumerate(orderedtimes)
    R = findall((out .>= tt) .& (in .< (tt-eps)) ) # risk set index (if in times are very close to other out-times, not using epsilon will make risk sets too big)
    ni = sum(wt[R]) # sum of weights in risk set
    di = sum(wt[R] .* (d[R] .> censval) .* (out[R] .== tt))
    _1mdovern[_i] = log(1.0 - di/ni)
    riskset[_i] = ni
  end
  orderedtimes, exp.(cumsum(_1mdovern)), riskset, [:times, :surv_overall, :riskset]
end


"""
Aalen-Johansen (cumulative incidence) with late entry, possibly multiple observations per unit
just a differently named function with more explicit output
"""
function aalen_johansen(in,out,d;dvalues=[1.0, 2.0], wt=nothing, eps = 0.00000001)
  if isnothing(wt) || isnan(wt[1])
    wt = ones(length(in))
  end
  nvals = length(dvalues)
  # overall survival via Kaplan-Meier
  orderedtimes, S, riskset, _ = kaplan_meier(in,out,d, wt=wt, eps=eps) # note ordered times are unique
  Sm1 = vcat(1.0, S)
  ajest = zeros(length(orderedtimes), nvals)
  _dij = zeros(length(orderedtimes), nvals)
  _d = zeros(length(out), nvals)
  @inbounds for (jidx,j) in enumerate(dvalues)
    _d[:,jidx] = (d .== j)
  end
  @inbounds for (_i,tt) in enumerate(orderedtimes)
    R = findall((out .>= tt) .& (in .< (tt-eps))) # risk set
    wtR = wt[R]
    ni = sum(wtR) # sum of weights/weighted individuals in risk set
    @inbounds for (jidx,j) in enumerate(dvalues)
        # = dij
        _dij[_i,jidx] = sum(wtR .* _d[R,jidx] .* (out[R] .== tt))
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
  using Distributions, Plots, BenchmarkTools
  plotly()
  z,x,t,d, event,wt = dgm_comprisk(n=10000, rng=MersenneTwister(1232));
  
  @btime times_sd, cumhaz, ci_sd = subdistribution_hazard_cuminc(zeros(length(t)), t, event, dvalues=[1.0, 2.0]);
  @btime times_aj, _, ajest, riskset, events = aalen_johansen(zeros(length(t)), t, event, dvalues=[1.0, 2.0]);
  
  plot(times_aj, ajest[:,1], label="AJ", st=:step);
  plot!(times_sd, ci_sd, label="SD", st=:step)  
  
  
"""
function subdistribution_hazard_cuminc(in,out,d;dvalues=[1.0, 2.0], wt=nothing, eps = 0.00000001)
  # ties allowed
  dmain = dvalues[1]
  if isnothing(wt) || isnan(wt[1])
    wt = ones(length(in))
  end
  censval = zero(eltype(d))
  times_dmain = unique(out[findall(d .== dmain)])
  orderedtimes_dmain = sort(times_dmain)
  #pseudo_riskset = zeros(Float64, length(times_dmain)) # risk set size
  #cases = zeros(length(orderedtimes_dmain))
  #_dt = zeros(length(orderedtimes))
  _haz = ones(length(orderedtimes_dmain))
  @inbounds for (_i,tt) in enumerate(orderedtimes_dmain)
    aliveandatriskidx = findall((in .< (tt-eps)) .&& (out .>= tt))
    hadcompidx = findall((out .<= tt) .&& (d .!= dmain))
    dmain_now = findall((out .== tt) .&& (d .== dmain))
    pseudoR = union(aliveandatriskidx, hadcompidx) # risk set index (if in times are very close to other out-times, not using epsilon will make risk sets too big)
    casesidx = intersect(dmain_now, aliveandatriskidx)
    ni = sum(wt[pseudoR]) # sum of weights in risk set
    di = sum(wt[casesidx])
    _haz[_i] = di/ni
    #cases[_i] = di
    #pseudo_riskset[_i] = ni
  end
  #orderedtimes_dmain, cumsum(_haz), 1.0 .- exp.(.-cumsum(_haz)), cases, pseudo_riskset, [:times, :cumhaz, :ci, :cases, :pseudo_riskset]
  orderedtimes_dmain, cumsum(_haz), 1.0 .- exp.(.-cumsum(_haz)), [:times, :cumhaz, :ci]
end
;

"""
Expected number of years of life lost due to cause k

  using Distributions, Plots, Random
  plotly()
  z,x,t,d, event,wt = dgm_comprisk(n=200, rng=MersenneTwister(1232));
  
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