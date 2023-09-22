```@meta
CurrentModule = LSurvival=
```

# [LSurvival](https://github.com/alexpkeil1/LSurvival.jl)

Survival analysis functions in Julia for time-to-event outcomes that can include:
- Loss-to-follow-up/right censoring
- Late entry/left truncation (not available in Survival.jl)
- "Person-period" data structures (not available in Survival.jl)
- Observation weights (not available in Survival.jl)
- Competing risks (not available in Survival.jl)

Capabilities include estimators for
- Kaplan-Meier non-parametric conditional risk functions
- Aalen-Johansen non-parametric cause-specific unconditional risk functions
- Cox proportional hazards model (Efron's or Breslow's methods for ties)

Convenience functions enable:
- Non-parametric bootstrapping, cluster-bootstrapping
- Estimating baseline hazards from a Cox Model
- Estimating cause-specific risk from an exhaustive set of Cox models for competing risk outcomes
- Simple simulation of competing and non-competing survival events
- Cluster robust variance estimation (without bootstrapping)
- Martingale, score, Schoenfeld, and dfbeta residuals
- Cluster robust variance estimation


Plans to include:
- Parametric survival models
- Stratification in Cox models

The package has been tuned to follow the "survival" package from R in terms of specific estimators/results.

Report issues [here](https://github.com/alexpkeil1/LSurvival.jl/issues)

## Installation 
```{julia}
using Pkg; Pkg.add(url = "https://github.com/alexpkeil1/LSurvival.jl")
```

## Quick examples

### Single event type: Cox model and Kaplan-Meier curve
```{julia}
using Random, LSurvival, Distributions, LinearAlgebra

# generate some data under a discrete hazards model
expit(mu) = inv(1.0 + exp(-mu))

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
            l = expit(-3 + 2 * v + 0 * l + 0 * a) > rand(rng) ? 1 : 0
            a = 0.1 > rand(rng) ? 1 : 0
            y = expit(-3 + 2 * v + 0 * l + 2 * a) > rand(rng) ? 1 : 0
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
wt = rand(length(d)) # random weights just to demonstrate usage

# Cox model
# Breslow's partial likelihood
m = fit(PHModel, X, int, outt, d, ties = "breslow", wts = wt)

# Efron's partial likelihood
m2 = fit(PHModel, X, int, outt, d, ties = "efron", wts = wt)

#equivalent way to specify 
# using `coxph` function
m2b = coxph(X, int, outt, d, ties = "efron", wts = wt)

# using `coxph` function with `Tables.jl` and `StatsAPI.@formula` interface (similar to GLM.jl)
tab = ( in = int, out = out, d=d, x=X[:,1], z1=X[:,2], z2=X[:,3]) # can also be a DataFrame from DataFrames.jl
m2b = coxph(@formula(Surv(in, out, d)~x+z1+z2), ties = "efron", wts = wt)

# can also be done if there is no late entry
m2b = coxph(@formula(Surv(out, d)~x+z1+z2), ties = "efron", wts = wt)
# can also be done if there is no late entry and no right censoring (i.e. all times are failure times)
m2b = coxph(@formula(Surv(out)~x+z1+z2), ties = "efron", wts = wt)



# Kaplan-Meier estimator of the cumulative risk/survival
res = kaplan_meier(int, outt, d)
```


### Competing event analysis: Aalen-Johansen and Cox-model-based estimators of the cumulative risk/survival
```{julia}
using Random, LSurvival, Distributions, LinearAlgebra

# simulate some data
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

z, x, t, d, event, wt = dgm_comprisk(; n = 100, rng = MersenneTwister(12))
X = hcat(x,z)
enter = t .* rand(length(d))*0.02 # create some fake entry times

# Aalen-Johansen estimator: marginal cause-specific risks
res_aj = aalen_johansen(enter, t, event; wts = wt);
res_aj

# Cox-model estimator: cause-specific risks at given levels of covariates
fit1 = fit(PHModel, X, enter, t, (event .== 1), ties = "efron",  wts = wt)
#n2idx = findall(event .!= 1)
n2idx = findall(event .> -1)
fit2 = fit(PHModel, X[n2idx,:], enter[n2idx], t[n2idx], (event[n2idx] .== 2), ties = "breslow",  wts = wt[n2idx])

# risk at referent levels of `x` and `z` (can be very extreme if referent levels are unlikely/unobservable)
res_cph_ref = risk_from_coxphmodels([fit1,fit2])

# risk at average levels of `x` and `z`
mnx = sum(x)/length(x)
mnz = sum(z)/length(z)
res_cph = risk_from_coxphmodels([fit1,fit2], coef_vectors=[coef(ft1), coef(ft2)], pred_profile=mean(X, dims=1))
# compare to Aalen-Johansen fit
res_aj


# this approach operates on left censored outcomes (which operate in the background in model fitting)
LSurvivalResp(enter, t, d, origintime=0)
LSurvivalCompResp(enter, t, event) # automatically infers origin


# can use the ID type to refer to units with multiple observations
id, int, outt, data = dgm(MersenneTwister(), 1000, 10; regimefun = int_0)
LSurvivalResp(int, outt, data[:,4], ID.(id))
```

## Index of functions

```@index
```

## Function help 

```@autodocs
Modules = [LSurvival]
```

```@contents
Pages = ["Likelihood.md"]
```