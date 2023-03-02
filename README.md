# LSurvival

[![Build Status](https://github.com/alexpkeil1/LSurvival.jl/actions/workflows/runtests.yml/badge.svg?branch=main)](https://github.com/alexpkeil1/LSurvival.jl/actions/workflows/runtests.yml?query=branch%3Amain)

These are some survival analysis functions that I was hoping to find in Julia and never did. They don't interface with the existing Julia model fitting modules (e.g. StatsModels). I needed a module that did these things, and I'm putting it here in case anyone is motivated to adapt this to fit better into the Julia ecosystem.

This module handles:
- Cox proportional hazards model with Efron's method or Breslow's method for ties
- Non-parametric survival function estimation via Kaplan-Meier
- Non-parametric survival function estimation under competing risks via Aalen-Johansen estimator
- Late entry/left trunctation and right censoring (all estimators)
- Baseline hazard estimator from a Cox model
- Cumulative incidence estimation from a set of Cox models

Add this package via Julia's package manager:

`add https://github.com/alexpkeil1/LSurvival.jl`

or directly in Julia

`using Pkg; Pkg.add(url="https://github.com/alexpkeil1/LSurvival.jl")`


```
using Random, LSurvival, Distributions, LinearAlgebra

# generate some data
  expit(mu) =  inv(1.0+exp(-mu))

  function int_nc(v,l,a)
    expit(-1.0 + 3*v + 2*l)
  end
  
  function int_0(v,l,a)
    0.1
  end
  
  function lprob(v,l,a)
    expit(-3 + 2*v + 0*l + 0*a)
  end

  function yprob(v,l,a)
    expit(-3 + 2*v + 0*l + 2*a)
  end

function dgm(rng, n, maxT;regimefun=int_0)
    V = rand(rng, n)
    LAY = Array{Float64,2}(undef,n*maxT,4)
    keep = ones(Bool, n*maxT)
    id = sort(reduce(vcat, fill(collect(1:n), maxT)))
    time = (reduce(vcat, fill(collect(1:maxT), n)))
    for i in 1:n
      v=V[i]; l = 0; a=0;
      lkeep = true
      for t in 1:maxT
          currIDX = (i-1)*maxT + t
          l = lprob(v,l,a) > rand(rng) ? 1 : 0
          a = regimefun(v,l,a) > rand(rng) ? 1 : 0
          y = yprob(v,l,a) > rand(rng) ? 1 : 0
          LAY[currIDX,:] .= [v,l,a,y]
          keep[currIDX] = lkeep
          lkeep = (!lkeep || (y==1)) ? false : true
      end
    end 
    id[findall(keep)], time[findall(keep)] .- 1, time[findall(keep)],LAY[findall(keep),:]
  end
  
  using Random
  id, int, outt, data = dgm(MersenneTwister(), 1000, 10;regimefun= int_0)
  data[:,1] = round.(  data[:,1] ,digits=3)
  d,X = data[:,4], data[:,1:3]
  wt = rand(length(d))

# Cox model
  coxargs = (int, outt, d, X);
  beta, ll, g, h, basehaz = coxmodel(coxargs..., weights=wt, method="breslow", tol=1e-9, inits=nothing);
  beta2, ll2, g2, h2, basehaz2 = coxmodel(coxargs..., weights=wt, method="efron", tol=1e-9, inits=nothing);
  se = sqrt.(diag(-inv(h)));
  hcat(beta, se, beta ./ se)

# Kaplan-Meier estimator of the cumulative risk/survival
res = kaplan_meier(int, outt,d; wt=nothing)
hcat(res[1], res[2])


# Competing risk analysis with Aalen-Johansen estimator of the cumulative risk/survival

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

z, x, t, d, event, wt = dgm_comprisk(;n=100, rng=MersenneTwister())

enter = t.*0.01

res = aalen_johansen(enter, t,event;dvalues=[1.0, 2.0], wt=wt)
hcat(res[1], res[3])

```
