# Parametric survival/risk estimation with Weibull AFT models

## Weibull accelerated failure time model
```julia
cd("docs/src/fig/")
using Random, LSurvival, Distributions, LinearAlgebra, Plots

# generate some data under a discrete hazards model
 id, int, out, data = LSurvival.dgm(MersenneTwister(1212), 1000, 20)

data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(MersenneTwister(1212), length(d)) # random weights just to demonstrate usage


# Fit a Cox model with `Tables.jl` and `StatsAPI.@formula` interface (similar to GLM.jl)
tab = (id=id, in = int, out = out, d=d, x=X[:,1], z1=X[:,2], z2=X[:,3], wts=wt) # can also be a DataFrame from DataFrames.jl
weibullfit = survreg(@formula(Surv(in, out, d)~x+z1+z2), tab, wts=tab.wts, dist=LSurvival.Weibull())
```

Output:

```output
Maximum likelihood estimates (alpha=0.05):
────────────────────────────────────────────────────────────────────────
                   Est     StdErr        LCI        UCI       Z  P(>|Z|)
────────────────────────────────────────────────────────────────────────
(Intercept)   2.92482   0.0808023   2.76645    3.08319    36.20   <1e-99
x            -1.50998   0.135432   -1.77543   -1.24454   -11.15   <1e-99
z1           -0.072211  0.106681   -0.281301   0.136879   -0.68   0.4985
z2           -1.08864   0.0906976  -1.2664    -0.910874  -12.00   <1e-99
log(Scale)   -0.223338  0.0368294  -0.295522  -0.151153   -6.06   <1e-08
────────────────────────────────────────────────────────────────────────
Weibull distribution
Log-likelihood (full): -1222.58
Log-likelihood (Intercept only):  -1356.2
LRT p-value (X^2=267.23, df=3): 0
Solver iterations: 15
```

## Comparing Weibull AFT and Cox model results

For the Weibull distribution, AFT model and Cox model results can be compared directly by converting AFT estimates to hazard ratios

```julia
coxfit = coxph(@formula(Surv(in, out, d)~x+z1+z2), tab, ties = "efron", wts = wt, id = ID.(tab.id))
```

Output:

```output
Maximum partial likelihood estimates (alpha=0.05):
───────────────────────────────────────────────────────────
      ln(HR)    StdErr        LCI       UCI      Z  P(>|Z|)
───────────────────────────────────────────────────────────
x   1.78123   0.181734   1.42504   2.13742    9.80   <1e-99
z1  0.109247  0.133533  -0.152473  0.370968   0.82   0.4133
z2  1.59741   0.10216    1.39718   1.79764   15.64   <1e-99
───────────────────────────────────────────────────────────
Partial log-likelihood (null): -2460.82
Partial log-likelihood (fitted): -2314.18
LRT p-value (X^2=293.28, df=3): 0
Newton-Raphson iterations: 6
```

Convert the AFT model parameters to hazard ratios to compare (note that this conversion is not possible for all parametric survival distributions).

```julia
scale = exp(weibullfit.P._S[1])
aftparms = coef(weibullfit)[2:end]
parmhrs = - aftparms ./ scale
hcat(coef(coxfit), parmhrs)
```

Output:
The first column is ln(HR) estimate from a Cox model, and the second is from the Weibull model

```output
3×2 Matrix{Float64}:
 1.78123   1.88785
 0.109247  0.0902812
 1.59741   1.36106
```



## Other distributions


### Exponential

```
expfit = survreg(@formula(Surv(in, out, d)~x+z1+z2), tab, wts=tab.wts, dist=LSurvival.Exponential())
```

Output:

```output
Maximum likelihood estimates (alpha=0.05):
──────────────────────────────────────────────────────────────────────
                   Est    StdErr        LCI       UCI       Z  P(>|Z|)
──────────────────────────────────────────────────────────────────────
(Intercept)   2.95353   0.100392   2.75677    3.15029   29.42   <1e-99
x            -1.55072   0.16808   -1.88015   -1.22129   -9.23   <1e-99
z1           -0.082459  0.133349  -0.343818   0.1789    -0.62   0.5363
z2           -1.36337   0.101441  -1.5622    -1.16455  -13.44   <1e-99
──────────────────────────────────────────────────────────────────────
Exponential distribution
Log-likelihood (full): -1239.09
Log-likelihood (Intercept only):  -1359.4
LRT p-value (X^2=240.61, df=3): 0
Solver iterations: 14
```


### Log-normal

```julia
# note this model runs into convergence issues in these data
    #expfit = survreg(@formula(Surv(in, out, d)~x+z1+z2), tab, wts=tab.wts, dist=LSurvival.Lognormal())

# Here are results from a simpler model
    dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])
    lognormalfit = survreg(@formula(Surv( time, status)~x), dat1, dist=LSurvival.Lognormal())
```

```output
Maximum likelihood estimates (alpha=0.05):
─────────────────────────────────────────────────────────────────────
                   Est    StdErr       LCI        UCI      Z  P(>|Z|)
─────────────────────────────────────────────────────────────────────
(Intercept)   2.20995   0.40358    1.41895   3.00095    5.48   <1e-07
x            -1.26752   0.585011  -2.41412  -0.120918  -2.17   0.0303
log(Scale)   -0.445615  0.342319  -1.11655   0.225318  -1.30   0.1930
─────────────────────────────────────────────────────────────────────
Lognormal distribution
Log-likelihood (full): -10.4662
Log-likelihood (Intercept only): -12.9106
LRT p-value (X^2=4.89, df=2): 0.086774
Solver iterations: 9
```


### Gamma
In progress

### Generalized gamma
In progress

### Log-logistic
In progress

### Gompertz
In progress