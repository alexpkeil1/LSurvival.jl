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
- Parametric survival models

Convenience functions enable:
- Non-parametric bootstrapping, cluster-bootstrapping, jackknife
- Estimating baseline hazards from a Cox Model
- Estimating cause-specific risk from an exhaustive set of Cox models for competing risk outcomes
- Simple simulation of competing and non-competing survival events
- Martingale, score, Schoenfeld, and dfbeta residuals
- Cluster robust variance estimation (Cox models)

Plans to include:
- Parametric survival models: more distributions
- Stratification in Cox models
- Parametric survival model diagnostics

The package has been tuned to follow the "survival" package from R in terms of specific estimators/results.

Report issues [here](https://github.com/alexpkeil1/LSurvival.jl/issues)

## Installation 
```{julia}
using Pkg; Pkg.add(url = "https://github.com/alexpkeil1/LSurvival.jl")
```

## Quick examples

### Single event type: Cox model and Kaplan-Meier curve
```julia

tab = ( in = int, out = out, d=d, x=X[:,1], z1=X[:,2], z2=X[:,3]) 

coxph(@formula(Surv(in, out, d)~x+z1+z2), tab, ties = "efron", wts = wt)
```

Output:
```
Maximum partial likelihood estimates (alpha=0.05):
─────────────────────────────────────────────────────────
      ln(HR)    StdErr        LCI      UCI     Z  P(>|Z|)
─────────────────────────────────────────────────────────
x   1.60222   0.530118   0.563208  2.64123  3.02   0.0025
z1  0.305929  0.43838   -0.55328   1.16514  0.70   0.4853
z2  1.98011   0.325314   1.34251   2.61771  6.09   <1e-08
─────────────────────────────────────────────────────────
Partial log-likelihood (null): -150.162
Partial log-likelihood (fitted): -131.512
LRT p-value (X^2=37.3, df=3): 3.9773e-08
Newton-Raphson iterations: 5
```

```julia
# can also be done if there is no late entry
coxph(@formula(Surv(out, d)~x+z1+z2), tab, ties = "efron", wts = wt)
# can also be done if there is no late entry and no right censoring (i.e. all times are failure times)
coxph(@formula(Surv(out)~x+z1+z2), tab, ties = "efron", wts = wt)

# Kaplan-Meier estimator of the cumulative risk/survival
kaplan_meier(int, outt, d)
```


### Competing event analysis: Aalen-Johansen and Cox-model-based estimators of the cumulative risk/survival

```julia
# Aalen-Johansen estimator: marginal cause-specific risks
res_aj = aalen_johansen(enter, t, event; wts = wt);
res_aj

# Cox-model estimator: cause-specific risks at given levels of covariates
fit1 = fit(PHModel, X, enter, t, (event .== 1), ties = "efron",  wts = wt)
n2idx = findall(event .!= 1)
fit2 = fit(PHModel, X[n2idx,:], enter[n2idx], t[n2idx], (event[n2idx] .== 2), ties = "efron",  wts = wt[n2idx])

# risk at average levels of `x` and `z`
res_cph = risk_from_coxphmodels([fit1,fit2], coef_vectors=[coef(fit1), coef(fit2)], pred_profile=mean(X, dims=1))
# compare to Aalen-Johansen fit
res_aj


# this approach operates on left censored outcomes (which operate in the background in model fitting)
LSurvivalResp(enter, t, d, origintime=0)
LSurvivalCompResp(enter, t, event) # automatically infers origin


# can use the ID type to refer to units with multiple observations
id, int, outt, data = dgm(MersenneTwister(), 1000, 10; regimefun = int_0)
LSurvivalResp(int, outt, data[:,4], ID.(id))
```

# Index of functions

```@index
```

# Function help 

```@autodocs
Modules = [LSurvival]
```

# Implementation details and further help

```@contents
Pages = [
    "Likelihood.md",
    "nonparametric.md",
    "coxmodel.md",
    "parametric.md",
    ]
    Depth = 3
```