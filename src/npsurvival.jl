


struct LSurvCompResp{E<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,W<:AbstractVector,T<:Real,V<:AbstractVector,M<:AbstractMatrix} <: AbstractLSurvResp
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
  ne = length(enter)
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
    wts = ones(Int64, ny)
  end
  eventtypes = sort(unique(y))
  eventmatrix = reduce(hcat, [y .== e for e in eventtypes[2:end]])

  return LSurvCompResp(enter, exit, y, wts, eventtimes, origin, eventtypes, eventmatrix)
end


mutable struct KMSurv{G<:LSurvResp} <: AbstractNPSurv
  R::G        # Survival response
  times::AbstractVector
  surv::Vector{Float64}
  riskset::Vector{Float64}
  events::Vector{Float64}
end

function KMSurv(R::G) where {G<:LSurvResp}
  times = R.eventtimes
  nt = length(times)
  surv = ones(Float64, nt)
  riskset = zeros(Float64, nt)
  events = zeros(Float64, nt)
  KMSurv(R, times, surv, riskset, events)
end

mutable struct AJSurv{G<:LSurvCompResp} <: AbstractNPSurv
  R::G        # Survival response
  times::AbstractVector
  surv::Vector{Float64}
  risk::Matrix{Float64}
  riskset::Vector{Float64}
  events::Matrix{Float64}
end

function AJSurv(R::G) where {G<:LSurvCompResp}
  times = R.eventtimes
  net = length(R.eventtypes) - 1
  nt = length(times)
  surv = ones(Float64, nt)
  risk = zeros(Float64, nt, net)
  riskset = zeros(Float64, nt)
  events = zeros(Float64, nt, net)
  AJSurv(R, times, surv, risk, riskset, events)
end


"""
   using LSurvival
   using Random
   z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 100);
   enter = zeros(length(t));
   X = hcat(x,z);
   R = LSurvResp(enter, t, Int64.(d), wt)
   mf = KMSurv(R)
   LSurvival._fit!(mf)

"""
function _fit!(m::KMSurv;
  eps=0.00000001,
  censval=0,
  kwargs...)
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
  m
end

#function aj(in,out,d;dvalues=[1.0, 2.0], weights=nothing, eps = 0.00000001)
function _fit!(m::AJSurv;
  #dvalues=[1.0, 2.0], 
  eps=0.00000001
)
  dvalues = m.R.eventtypes[2:end]
  nvals = length(dvalues)
  kmfit = fit(KMSurv, m.R.enter, m.R.exit, m.R.y, weights=m.R.wts)
  m.surv = kmfit.surv
  # overall survival via Kaplan-Meier
  orderedtimes, S, riskset = kmfit.times, kmfit.surv, kmfit.riskset
  Sm1 = vcat(1.0, S)
  #####
  #_d = zeros(length(m.R.exit), nvals)
  #for (jidx,j) in enumerate(dvalues)
  #  _d[:,jidx] = (d .== j)
  #end
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
  for jidx in 1:nvals
    m.risk[:, jidx] = cumsum(m.risk[:, jidx])
  end
  m
  #orderedtimes, S, ajest, riskset
end
;


function StatsBase.fit!(m::AbstractNPSurv;
  kwargs...)
  _fit!(m; kwargs...)
end

"""
fit for KMSurv objects

   using LSurvival
   using Random
   z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
   enter = zeros(length(t));
   m = fit(KMSurv, enter, t, d)
   mw = fit(KMSurv, enter, t, d, wts=wt)
  """
function fit(::Type{M},
  enter::AbstractVector{<:Real},
  exit::AbstractVector{<:Real},
  y::Union{AbstractVector{<:Real},BitVector}
  ;
  wts::AbstractVector{<:Real}=similar(y, 0),
  offset::AbstractVector{<:Real}=similar(y, 0),
  fitargs...) where {M<:KMSurv}

  R = LSurvResp(enter, exit, y, wts)
  res = M(R)

  return fit!(res; fitargs...)
end

"""
fit for AJSurv objects

   using LSurvival
   using Random
   z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
   enter = zeros(length(t));
   m = fit(AJSurv, enter, t, d)
   mw = fit(AJSurv, enter, t, d, wts=wt)
  """
function fit(::Type{M},
  enter::AbstractVector{<:Real},
  exit::AbstractVector{<:Real},
  y::Union{AbstractVector{<:Real},BitVector}
  ;
  wts::AbstractVector{<:Real}=similar(y, 0),
  offset::AbstractVector{<:Real}=similar(y, 0),
  fitargs...) where {M<:AJSurv}

  R = LSurvCompResp(enter, exit, y, wts)
  res = M(R)

  return fit!(res; fitargs...)
end

"""
  kaplan_meier(enter::AbstractVector, exit::AbstractVector, y::AbstractVector,
      ; <keyword arguments>)
"""
kaplan_meier(enter, exit, y, args...; kwargs...) = fit(KMSurv, enter, exit, y, args...; kwargs...)

"""
  aalen_johansen(enter::AbstractVector, exit::AbstractVector, d::AbstractVector,
      ; <keyword arguments>)
"""
aalen_johansen(enter, exit, d, args...; kwargs...) = fit(AJSurv, enter, exit, d, args...; kwargs...)


function Base.show(io::IO, m::M; maxrows=20) where {M<:KMSurv}
  resmat = hcat(m.times, m.surv, m.events, m.riskset)
  head = ["time", "survival", "# events", "at risk"]
  nr = size(resmat)[1]
  rown = ["$i" for i in 1:nr]
  op = CoefTable(resmat, head, rown)
  iob = IOBuffer()
  if nr < maxrows
    println(iob, op)
  else
    len = round(Int, maxrows / 2)
    op1, op2 = deepcopy(op), deepcopy(op)
    op1.rownms = op1.rownms[1:len]
    op1.cols = [c[1:len] for c in op1.cols]
    op2.rownms = op2.rownms[(end-len):end]
    op2.cols = [c[(end-len):end] for c in op2.cols]
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


function Base.show(io::IO, m::M; maxrows=20) where {M<:AJSurv}
  types = m.R.eventtypes[2:end]
  ev = ["# events (j=$jidx)" for (jidx, j) in enumerate(types)]
  rr = ["risk (j=$jidx)" for (jidx, j) in enumerate(types)]

  resmat = hcat(m.times, m.surv, m.events, m.riskset, m.risk)
  head = ["time", "survival", ev..., "at risk", rr...]
  nr = size(resmat)[1]
  rown = ["$i" for i in 1:nr]

  op = CoefTable(resmat, head, rown)
  iob = IOBuffer()
  if nr < maxrows
    println(iob, op)
  else
    len = round(Int, maxrows / 2)
    op1, op2 = deepcopy(op), deepcopy(op)
    op1.rownms = op1.rownms[1:len]
    op1.cols = [c[1:len] for c in op1.cols]
    op2.rownms = op2.rownms[(end-len):end]
    op2.cols = [c[(end-len):end] for c in op2.cols]
    println(iob, op1)
    println(iob, "...")
    println(iob, op2)
  end
  str = """\nKaplan-Meier Survival, Aalen-Johansen risk\n"""
  str *= String(take!(iob))
  for (jidx, j) in enumerate(types)
    str *= "Number of events (j=$j): $(@sprintf("%8g", sum(m.events[:,jidx])))\n"
  end
  str *= "Number of unique event times: $(@sprintf("%8g", length(m.events)))\n"
  println(io, str)
end



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
function e_yearsoflifelost(time, ci)
  cilag = vcat(0, ci[1:(end-1)])
  difftime = diff(vcat(0, time))
  time, cumsum(cilag .* difftime)
end


;