```@meta
CurrentModule = LSurvival=
```

# [LSurvival](https://github.com/alexpkeil1/LSurvival.jl).

Survival analysis functions in Julia for time-to-event outcomes that can include:
- Loss-to-follow-up/right censoring
- Late entry/left truncation (not available in the Survival.jl package)
- "Person-period" data structures (not available in the Survival.jl package)
- Observation weights (not available in the Survival.jl package)
- Competing risks (not available in the Survival.jl package)

Capabilities include estimators for
- Kaplan-Meier non-parametric conditional risk functions
- Aalen-Johansen non-parametric cause-specific unconditional risk functions
- Cox proportional hazards model (Efron's or Breslow's methods for ties)

Convenience functions enable:
- Non-parametric bootstrapping, cluster-bootstrapping
- Estimating baseline hazards from a Cox Model
- Estimating cause-specific risk from an exhaustive set of Cox models for competing risk outcomes
- Simple simulation of competing and non-competing survival events

Plans to include:
- cluster robust variance estimation (without bootstrapping)

Report issues [here](https://github.com/alexpkeil1/LSurvival.jl/issues)

## Installation 
```{julia}
using Pkg; Pkg.add(url = "https://github.com/alexpkeil1/LSurvival.jl")
```

## Quick examples
```{julia}
using Random, LSurvival, Distributions, LinearAlgebra

# generate some data
expit(mu) = inv(1.0 + exp(-mu))

function int_nc(v, l, a)
    expit(-1.0 + 3 * v + 2 * l)
end

function int_0(v, l, a)
    0.1
end

function lprob(v, l, a)
    expit(-3 + 2 * v + 0 * l + 0 * a)
end

function yprob(v, l, a)
    expit(-3 + 2 * v + 0 * l + 2 * a)
end

function dgm(rng, n, maxT; regimefun = int_0)
    V = rand(rng, n)
    LAY = Array{Float64,2}(undef, n * maxT, 4)
    keep = ones(Bool, n * maxT)
    id = sort(reduce(vcat, fill(collect(1:n), maxT)))
    time = (reduce(vcat, fill(collect(1:maxT), n)))
    for i = 1:n
        v = V[i]
        l = 0
        a = 0
        lkeep = true
        for t = 1:maxT
            currIDX = (i - 1) * maxT + t
            l = lprob(v, l, a) > rand(rng) ? 1 : 0
            a = regimefun(v, l, a) > rand(rng) ? 1 : 0
            y = yprob(v, l, a) > rand(rng) ? 1 : 0
            LAY[currIDX, :] .= [v, l, a, y]
            keep[currIDX] = lkeep
            lkeep = (!lkeep || (y == 1)) ? false : true
        end
    end
    id[findall(keep)], time[findall(keep)] .- 1, time[findall(keep)], LAY[findall(keep), :]
end

id, int, outt, data = dgm(MersenneTwister(), 1000, 10; regimefun = int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))

# Cox model
m = fit(PHModel, X, int, outt, d, ties = "breslow", wts = wt)
m2 = fit(PHModel, X, int, outt, d, ties = "efron", wts = wt)
#equivalent
m2b = coxph(X, int, outt, d, ties = "efron", wts = wt)

# Kaplan-Meier estimator of the cumulative risk/survival
res = kaplan_meier(int, outt, d)

# Competing risk analysis with Aalen-Johansen estimator of the cumulative risk/survival

function dgm_comprisk(; n = 100, rng = MersenneTwister())
    z = rand(rng, n) .* 5
    x = rand(rng, n) .* 5
    dt1 = Weibull.(fill(0.75, n), inv.(exp.(-x .- z)))
    dt2 = Weibull.(fill(0.75, n), inv.(exp.(-x .- z)))
    t01 = rand.(rng, dt1)
    t02 = rand.(rng, dt2)
    t0 = min.(t01, t02)
    t = Array{Float64,1}(undef, n)
    for i = 1:n
        t[i] = t0[i] > 1.0 ? 1.0 : t0[i]
    end
    d = (t .== t0)
    event = (t .== t01) .+ 2.0 .* (t .== t02)
    wtu = rand(rng, n) .* 5.0
    wt = wtu ./ mean(wtu)
    reshape(round.(z, digits = 4), (n, 1)),
    reshape(round.(x, digits = 4), (n, 1)),
    round.(t, digits = 4),
    d,
    event,
    round.(wt, digits = 4)
end

z, x, t, d, event, wt = dgm_comprisk(; n = 100, rng = MersenneTwister())
X = hcat(x,z)
enter = t .* rand(100)*0.02 # create some fake entry times

res = aalen_johansen(enter, t, event; wts = wt)
fit1 = fit(PHModel, X, enter, t, (event .== 1), ties = "breslow", wts = wt)
fit2 = fit(PHModel, X, enter, t, (event .== 1), ties = "efron", wts = wt)
risk_from_coxphmodels([fit1, fit2])

# this approach operates on left censored outcomes (which operate in the background in model fitting)
LSurvResp(enter, t, d)
LSurvCompResp(enter, t, event)


# can use the ID type to refer to units with multiple observations
id, int, outt, data = dgm(MersenneTwister(), 1000, 10; regimefun = int_0)
LSurvResp(int, outt, data[:,4], ID.(id))
```

## Index of functions

```@index
```

## Function help 

```@autodocs
Modules = [LSurvival]
```
