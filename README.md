# LSurvival

[![Build Status](https://github.com/alexpkeil1/LSurvival.jl/actions/workflows/runtests.yml/badge.svg?branch=main)](https://github.com/alexpkeil1/LSurvival.jl/actions/workflows/runtests.yml?query=branch%3Amain)
[![Main](https://img.shields.io/badge/docs-stable-blue.svg)](https://alexpkeil1.github.io/LSurvival.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-latest-blue.svg)](https://alexpkeil1.github.io/LSurvival.jl/dev/)
[![codecov](https://codecov.io/github/alexpkeil1/LSurvival.jl/graph/badge.svg?token=2LFIITQ2LV)](https://codecov.io/github/alexpkeil1/LSurvival.jl)

These are some survival analysis functions that I was hoping to find in Julia and never did. Interface with StatsModels is developing. I needed a module that did these things.

This module handles:
- Cox proportional hazards model with Efron's method or Breslow's method for ties
- Parametric accelerated failure time (AFT) models
- Non-parametric survival function estimation via Kaplan-Meier
- Non-parametric survival function estimation under competing risks via Aalen-Johansen estimator
- Late entry/left trunctation and right censoring (all estimators)
- Baseline hazard estimator from a Cox model
- Cumulative incidence estimation from a set of Cox models
- Cox model residuals (Martingale, score, Schoenfeld, dfbeta)
- Robust variance estimation
- Bootstrap and jackknife methods for Cox models and AFT models
- Plot recipes for diagnostics and visualizations

Add this package via Julia's package manager:

`add https://github.com/alexpkeil1/LSurvival.jl`

or directly in Julia

`using Pkg; Pkg.add(url="https://github.com/alexpkeil1/LSurvival.jl")`



## Cox model

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

# using the formula interface
tab = (
    entertime = int,
    exittime = outt,
    death = data[:, 4] .== 1,
    x = data[:, 1],
    z1 = data[:, 2],
    z2 = data[:, 3],
)
m2c = coxph(@formula(Surv(entertime, exittime, death) ~ x + z1 + z2), tab, wts = wt)
m2d = fit(PHModel, @formula(Surv(entertime, exittime, death) ~ x + z1 + z2), tab, wts = wt)

```

## Kaplan-Meier estimator of the cumulative risk/survival
```{julia}
res = kaplan_meier(int, outt, d)
confint(res, method="lognlog")
plot(res)
```

## Competing risk analysis with Aalen-Johansen estimator of the cumulative risk/survival

```{julia}
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
X = hcat(x, z)
enter = t .* rand(100) * 0.02 # create some fake entry times

res = aalen_johansen(enter, t, event; wts = wt)
fit1 = fit(PHModel, X, enter, t, (event .== 1), ties = "breslow", wts = wt)
fit2 = fit(PHModel, X, enter, t, (event .== 2), ties = "breslow", wts = wt)
risk_from_coxphmodels([fit1, fit2])

# this approach operates on left truncated outcomes (which operate in the background in model fitting)
LSurvivalResp(enter, t, d)
LSurvivalCompResp(enter, t, event)


# can use the ID type to refer to units with multiple observations
id, int, outt, data = dgm(MersenneTwister(), 1000, 10; regimefun = int_0)
LSurvivalResp(int, outt, data[:, 4], ID.(id))
```

More documentation can be found here: [stable version](https://alexpkeil1.github.io/LSurvival.jl/stable/), [developmental version](https://alexpkeil1.github.io/LSurvival.jl/dev/)

## Note about correctness

Where possible, results from this package have been validated by comparing results with the `survival` package in `R`, by Terry Therneau. Validation, in this case, requires some choices that may result in differences from the `PHREG` procedure in `SAS` or other software packages for Cox models due to differing approaches to, for example, weighting in the case of ties or the calculation of Martingale residuals under Efron's partial likelihood. See the `survival` package [vignette](https://cran.r-project.org/web/packages/survival/vignettes/validate.pdf) for some details that form the basis of module testing against answers that can be confirmed via laborious hand calculations.
