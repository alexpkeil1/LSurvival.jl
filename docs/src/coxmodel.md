# Cox models

```julia
cd("docs/src/fig/")
using Random, LSurvival, Distributions, LinearAlgebra, Plots, DataFrames

# generate some data under a discrete hazards model
 id, int, out, data = LSurvival.dgm(MersenneTwister(1212), 100, 20)

data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = ones(length(d)) # random weights just to demonstrate usage


# Fit a Cox model with `Tables.jl` and `StatsAPI.@formula` interface (similar to GLM.jl)
tab = (id=id, in = int, out = out, d=d, x=X[:,1], z1=X[:,2], z2=X[:,3]) # can also be a DataFrame from DataFrames.jl
df = DataFrame("id" => id, "x"=>X[:,1],"z"=>X[:,2],"t"=>out,"enter"=>int,"d"=>d,"wt"=>wt)
show(df)
```

Note here that covariates are not time-varying, but that person-period data structures are used (which could accomdate time-varying exposures). 
```output
634×7 DataFrame
 Row │ id     x        z        t      enter  d        wt      
     │ Int64  Float64  Float64  Int64  Int64  Float64  Float64 
─────┼─────────────────────────────────────────────────────────
   1 │     1    0.125      0.0      1      0      0.0      1.0
   2 │     1    0.125      0.0      2      1      0.0      1.0
   3 │     1    0.125      0.0      3      2      0.0      1.0
   4 │     1    0.125      0.0      4      3      0.0      1.0
  ⋮  │   ⋮       ⋮        ⋮       ⋮      ⋮       ⋮        ⋮
 631 │    99    0.41       0.0      1      0      1.0      1.0
 632 │   100    0.103      0.0      1      0      0.0      1.0
 633 │   100    0.103      0.0      2      1      0.0      1.0
 634 │   100    0.103      0.0      3      2      1.0      1.0
```

Note use of the `id` argument to specify that multiple observations come from the same individual. This is important in the case of robust-variance estimation, jackknifing, bootstrapping, and influence-based residuals like `dfbeta` residuals. It will have no impact on the default output (confirm by fitting this model with and without the statement, and also compare influence plots below)!

```julia
mfit = coxph(@formula(Surv(in, out, d)~x+z1+z2), tab, ties = "efron", wts = wt, id = ID.(tab.id))
```

Output:

```output
Maximum partial likelihood estimates (alpha=0.05):
─────────────────────────────────────────────────────────
     ln(HR)    StdErr        LCI       UCI     Z  P(>|Z|)
─────────────────────────────────────────────────────────
x   1.6289   0.385794   0.872755  2.38504   4.22   <1e-04
z1  0.16381  0.30964   -0.443074  0.770694  0.53   0.5968
z2  1.79485  0.238453   1.32749   2.26221   7.53   <1e-13
─────────────────────────────────────────────────────────
Partial log-likelihood (null): -353.135
Partial log-likelihood (fitted): -322.353
LRT p-value (X^2=61.56, df=3): 2.7234e-13
Newton-Raphson iterations: 6
```

## Plotting survival outcomes (person-period plot)

```julia
plot(mfit.R)
savefig("ppplot.svg")
```
![Person-period plot](fig/ppplot.svg)


## Estimating baseline hazards

Baseline hazards (at referent levels of covariates) are estimated by default in coxph.

```julia
mfit.bh
basehazplot(mfit)
savefig("basehaz.svg")
```
![Baseline hazard](fig/basehaz.svg)

## Model fit: Schoenfeld residuals

```julia
residuals(mfit, type="schoenfeld")
coxdx(mfit)
savefig("schoenfeld.svg")
```
![Schoenfeld](fig/schoenfeld.svg)

## Influence: Jackknife/dfbeta residuals
These will be on the individual level

```julia
residuals(mfit, type="dfbeta")
residuals(mfit, type="jackknife")

coxinfluence(mfit, type="jackknife", par=1)
coxinfluence!(mfit, type="dfbeta", color=:red, par=1)
savefig("influence.svg")
```
![Influence](fig/influence.svg)


## Competing event analysis: Cox-model-based estimator of the cumulative risk/survival function

```julia
using Random, LSurvival, Distributions, LinearAlgebra

# simulate some data and store in a DataFrame
using DataFrames
z, x, t, d, event, wt = LSurvival.dgm_comprisk(MersenneTwister(122), 1000)
X = hcat(x,z)
enter = t .* rand(length(d))*0.02 # create some fake entry times
df = DataFrame("x"=>x[:,1],"z"=>z[:,1],"t"=>t,"enter"=>enter,"event"=>event,"wt"=>wt)
show(df)
```

`event` can be 0 (censored) 1 (event type 1: e.g. death from lung cancer) or 2 (event type 2: e.g. death from causes other than lung cancer)

```output
1000×6 DataFrame
  Row │ x        z        t        enter       event    wt      
      │ Float64  Float64  Float64  Float64     Float64  Float64 
──────┼─────────────────────────────────────────────────────────
    1 │  2.7596   0.475    1.0     0.0168423       0.0   1.2425
    2 │  4.166    0.3008   1.0     0.00389611      0.0   0.5362
    3 │  1.1702   4.1267   1.0     0.0128694       0.0   0.4096
    4 │  2.7756   4.7509   1.0     0.0143146       0.0   0.5082
  ⋮   │    ⋮        ⋮        ⋮         ⋮          ⋮        ⋮
  997 │  4.3977   4.611    1.0     0.00917689      0.0   0.6857
  998 │  0.3136   1.8464   0.6908  0.0133034       2.0   1.118
  999 │  3.7132   3.7309   1.0     0.0106726       0.0   1.2687
 1000 │  0.9304   3.3816   1.0     0.0104032       0.0   0.4498
```

### Fitting cause-specific Cox models for competing event types

```julia
fit1 = coxph(@formula(Surv(enter, t, event==1)~x+z), df, wts=df.wt)
n2idx = findall(event .!= 1)
fit2 = coxph(@formula(Surv(enter, t, event==2)~x+z), df[n2idx,:], wts=df.wt[n2idx])
```

Fit, cause 1:

```output
Maximum partial likelihood estimates (alpha=0.05):
────────────────────────────────────────────────────────────
      ln(HR)    StdErr        LCI        UCI      Z  P(>|Z|)
────────────────────────────────────────────────────────────
x  -0.65619   0.123881  -0.898992  -0.413388  -5.30   <1e-06
z  -0.461087  0.11363   -0.683798  -0.238376  -4.06   <1e-04
────────────────────────────────────────────────────────────
Partial log-likelihood (null): -320.077
Partial log-likelihood (fitted): -294.521
LRT p-value (X^2=51.11, df=2): 7.9665e-12
Newton-Raphson iterations: 4

```

Fit, cause 2:

```output
Maximum partial likelihood estimates (alpha=0.05):
────────────────────────────────────────────────────────────
      ln(HR)    StdErr        LCI        UCI      Z  P(>|Z|)
────────────────────────────────────────────────────────────
x  -0.598941  0.104098  -0.802969  -0.394914  -5.75   <1e-08
z  -0.855451  0.123795  -1.09809   -0.612817  -6.91   <1e-11
────────────────────────────────────────────────────────────
Partial log-likelihood (null): -396.624
Partial log-likelihood (fitted): -348.373
LRT p-value (X^2=96.5, df=2): 0
Newton-Raphson iterations: 5

```

### Cox-model estimator: cause-specific risks at given levels of covariates

Risk at referent levels of `x` and `z` (can be very extreme if referent levels are unlikely/unobservable). E.g. 20% survival is very low, considering the kaplan-meier overall survival estimate at the end of follow-up is 88%. This illustrates that lower levels of `x` and `z` confer exceedingly high risks in this example, but the referent levels of x=0 and z=0 are not actually observed in the data. One could center these variables in the model fit or use the approach below of predicting risk at specific, non-referent values of `x` and `z`.

```julia
println("extrema: x")
extrema(x)
println("extrema: z")
extrema(z)
kaplan_meier(enter, t, d)
res_cph_ref = risk_from_coxphmodels([fit1,fit2])
```

```output
extrema: x
(0.0121, 4.997)

extrema: z
(0.0164, 4.9996)

Kaplan-Meier Survival
───────────────────────────────────────
      time  survival  # events  at risk
───────────────────────────────────────
1   0.0022  0.992806       1.0    139.0
2   0.0038  0.988703       1.0    242.0
3   0.0054  0.985504       1.0    309.0
4   0.0111  0.983793       1.0    576.0
5   0.0174  0.982688       1.0    891.0
6   0.0254  0.981701       1.0    995.0
7   0.0288  0.980713       1.0    994.0
8   0.0298  0.979726       1.0    993.0
9   0.0595  0.978738       1.0    992.0
10  0.061   0.97775        1.0    991.0
───────────────────────────────────────
...
────────────────────────────────────────
       time  survival  # events  at risk
────────────────────────────────────────
97   0.7976  0.890839       1.0    903.0
98   0.798   0.889852       1.0    902.0
99   0.8072  0.888864       1.0    901.0
100  0.815   0.887876       1.0    900.0
101  0.8174  0.886889       1.0    899.0
102  0.8309  0.885901       1.0    898.0
103  0.8386  0.884913       1.0    897.0
104  0.8572  0.883926       1.0    896.0
105  0.9051  0.882938       1.0    895.0
106  0.9189  0.881951       1.0    894.0
────────────────────────────────────────
Number of events:      107
Number of unique event times:      106

Cox-model based survival, risk, baseline cause-specific hazard
───────────────────────────────────────────────────────────────────────────────
      time  survival  event type  cause-specific hazard  risk (j=1)  risk (j=2)
───────────────────────────────────────────────────────────────────────────────
1   0.0022  0.992917         1.0             0.0070832    0.0070832   0.0
2   0.0038  0.976445         1.0             0.0165889    0.0235546   0.0
3   0.0054  0.906798         2.0             0.0713276    0.0235546   0.0696475
4   0.0111  0.896455         2.0             0.0114065    0.0235546   0.0799909
5   0.0174  0.889744         1.0             0.00748562   0.0302651   0.0799909
6   0.0254  0.865023         2.0             0.0277843    0.0302651   0.104712
7   0.0288  0.862307         1.0             0.00313978   0.0329811   0.104712
8   0.0298  0.859912         2.0             0.00277794   0.0329811   0.107107
9   0.0595  0.846608         1.0             0.0154706    0.0462844   0.107107
10  0.061   0.841296         1.0             0.00627444   0.0515964   0.107107
───────────────────────────────────────────────────────────────────────────────
...
────────────────────────────────────────────────────────────────────────────────
       time  survival  event type  cause-specific hazard  risk (j=1)  risk (j=2)
────────────────────────────────────────────────────────────────────────────────
97   0.7976  0.236682         2.0             0.0370947     0.250571    0.512747
98   0.798   0.232785         1.0             0.0164669     0.254468    0.512747
99   0.8072  0.229551         1.0             0.0138927     0.257702    0.512747
100  0.815   0.229253         1.0             0.00129742    0.258       0.512747
101  0.8174  0.22851          1.0             0.00324161    0.258743    0.512747
102  0.8309  0.223335         2.0             0.0226469     0.258743    0.517922
103  0.8386  0.214633         2.0             0.0389631     0.258743    0.526624
104  0.8572  0.213611         1.0             0.00475952    0.259764    0.526624
105  0.9051  0.211151         1.0             0.0115197     0.262225    0.526624
106  0.9189  0.210345         1.0             0.00381603    0.263031    0.526624
────────────────────────────────────────────────────────────────────────────────
Number of events (j=1):       52
Number of events (j=2):       54
Number of unique event times:      106
```

You can also estimate risk at average levels of `x` and `z` (or any level). Here, the survival (94%) is higher than the marginal survival of 88%, emphasizing that predicted risk at population average levels of covariates (the approach taken with Cox models here) can be different from population average risk across all levels of covariates (Kaplan-Meier).

```julia

mnx = sum(x)/length(x)
mnz = sum(z)/length(z)
res_cph = risk_from_coxphmodels([fit1,fit2], coef_vectors=[coef(fit1), coef(fit2)], pred_profile=mean(X, dims=1))
```


```output
Cox-model based survival, risk, baseline cause-specific hazard
────────────────────────────────────────────────────────────────────────────────
      time  survival  event type  cause-specific hazard   risk (j=1)  risk (j=2)
────────────────────────────────────────────────────────────────────────────────
1   0.0022  0.999559         1.0            0.000440709  0.000440709  0.0
2   0.0038  0.998528         1.0            0.00103214   0.0014724    0.0
3   0.0054  0.996639         2.0            0.00189161   0.0014724    0.00188883
4   0.0111  0.996337         2.0            0.000302502  0.0014724    0.00219031
5   0.0174  0.995873         1.0            0.000465747  0.00193644   0.00219031
6   0.0254  0.995139         2.0            0.000736841  0.00193644   0.00292411
7   0.0288  0.994945         1.0            0.000195354  0.00213084   0.00292411
8   0.0298  0.994872         2.0            7.3671e-5    0.00213084   0.00299741
9   0.0595  0.993914         1.0            0.000962562  0.00308847   0.00299741
10  0.061   0.993526         1.0            0.000390389  0.00347648   0.00299741
────────────────────────────────────────────────────────────────────────────────
...
────────────────────────────────────────────────────────────────────────────────
       time  survival  event type  cause-specific hazard  risk (j=1)  risk (j=2)
────────────────────────────────────────────────────────────────────────────────
97   0.7976  0.948754         2.0            0.000983755   0.0252707   0.0259756
98   0.798   0.947782         1.0            0.00102455    0.0262427   0.0259756
99   0.8072  0.946962         1.0            0.000864389   0.027062    0.0259756
100  0.815   0.946886         1.0            8.07241e-5    0.0271384   0.0259756
101  0.8174  0.946695         1.0            0.00020169    0.0273294   0.0259756
102  0.8309  0.946126         2.0            0.000600597   0.0273294   0.0265441
103  0.8386  0.945149         2.0            0.0010333     0.0273294   0.0275218
104  0.8572  0.944869         1.0            0.000296132   0.0276093   0.0275218
105  0.9051  0.944192         1.0            0.000716743   0.0282865   0.0275218
106  0.9189  0.943968         1.0            0.000237429   0.0285107   0.0275218
────────────────────────────────────────────────────────────────────────────────
Number of events (j=1):       52
Number of events (j=2):       54
Number of unique event times:      106
```


```julia
plot(res_cph)
savefig("risk-multicox.svg")
```
![Risk from competing risk Cox models](fig/risk-multicox.svg)

Here is another way to get risk at the reference level of `x` and `z`, more explicitly:

```julia
res_cph_ref = risk_from_coxphmodels([fit1,fit2], coef_vectors=[coef(fit1), coef(fit2)], pred_profile=[0.0, 0.0])
```


```julia
plot(res_cph_ref)
savefig("risk-multicox2.svg")
```
![Risk from competing risk Cox models](fig/risk-multicox2.svg)

The default uses an Aalen-Johansen analogue to estimate cumulative risks. These can occasionally result in risks outside of logical bounds. An alternative is the Cheng-Fine-Wei approach[^cfw], which uses the cumulative hazard to estimate survival via $S(t) = exp(-\Lambda(t))$. Here, you can see there is a negligible difference in the risk estimates between these two approaches (contrasted with the `res_cph_ref` object).

```julia
res_cph_ref_cheng = risk_from_coxphmodels([fit1,fit2], coef_vectors=[coef(fit1), coef(fit2)], pred_profile=[0.0, 0.0], method="cheng-fine-wei")
show(res_cph_ref, maxrows=4)
show(res_cph_ref_cheng, maxrows=4)

```

```output
# Aalen-Johansen method
Cox-model based survival, risk, baseline cause-specific hazard
──────────────────────────────────────────────────────────────────────────────
     time  survival  event type  cause-specific hazard  risk (j=1)  risk (j=2)
──────────────────────────────────────────────────────────────────────────────
1  0.0022  0.992917         1.0              0.0070832   0.0070832         0.0
2  0.0038  0.976445         1.0              0.0165889   0.0235546         0.0
──────────────────────────────────────────────────────────────────────────────
...
────────────────────────────────────────────────────────────────────────────────
       time  survival  event type  cause-specific hazard  risk (j=1)  risk (j=2)
────────────────────────────────────────────────────────────────────────────────
105  0.9051  0.211151         1.0             0.0115197     0.262225    0.526624
106  0.9189  0.210345         1.0             0.00381603    0.263031    0.526624
────────────────────────────────────────────────────────────────────────────────
Number of events (j=1):       52
Number of events (j=2):       54
Number of unique event times:      106

# Cheng, Fine, Wei method
Cox-model based survival, risk, baseline cause-specific hazard
──────────────────────────────────────────────────────────────────────────────
     time  survival  event type  cause-specific hazard  risk (j=1)  risk (j=2)
──────────────────────────────────────────────────────────────────────────────
1  0.0022  0.992942         1.0              0.0070832   0.0070832         0.0
2  0.0038  0.976606         1.0              0.0165889   0.023555          0.0
──────────────────────────────────────────────────────────────────────────────
...
────────────────────────────────────────────────────────────────────────────────
       time  survival  event type  cause-specific hazard  risk (j=1)  risk (j=2)
────────────────────────────────────────────────────────────────────────────────
105  0.9051  0.215072         1.0             0.0115197     0.263869     0.53052
106  0.9189  0.214253         1.0             0.00381603    0.26469      0.53052
────────────────────────────────────────────────────────────────────────────────
Number of events (j=1):       52
Number of events (j=2):       54
Number of unique event times:      106
```


### Cox-model estimator: standard errors and confidence intervals
Robust, jackknife (leave-on-out), and bootstrap standard errors are easily calculated from cox model fits (though they may be computationally demanding to calculate). For person-period data, each of these requires the `id` argument to be specified.

```julia
coxfit = coxph(@formula(Surv(in, out, d)~x+z1+z2), tab, id=ID.(tab.id))
show(coxfit)

asym = stderror(coxfit)   # asymptotic approach based on information matrix
rob = stderror(coxfit, type="robust")  # jackknife approach
jack = stderror(coxfit, type="jackknife")  # jackknife approach
boot = stderror(coxfit, type="bootstrap", iter=200, seed=MersenneTwister(12322)) # bootstrapping approach
DataFrame("asym" => asym, "rob" => rob,"jack" => jack, "boot" => boot)
```
We can see in this dataset all standard error estimates are fairly similar

```output
3×4 DataFrame
 Row │ asym      rob       jack      boot     
     │ Float64   Float64   Float64   Float64  
─────┼────────────────────────────────────────
   1 │ 0.385794  0.34421   0.37397   0.365392
   2 │ 0.30964   0.307867  0.326048  0.331512
   3 │ 0.238453  0.238146  0.260687  0.240776
```

These methods are also available for `vcov` and `confint`, where confint uses variance estimates to create Wald-type confidence intervals (i.e. not bootstrap percentile confidence intervals)

```julia
asym = confint(coxfit)   # asymptotic approach based on information matrix
rob = confint(coxfit, type="robust")  # jackknife approach
jack = confint(coxfit, type="jackknife")  # jackknife approach
boot = confint(coxfit, type="bootstrap", iter=200, seed=MersenneTwister(12322)) # bootstrapping approach
DataFrame("asym" => asym[:,1], "rob" => rob[:,1],"jack" => jack[:,1], "boot" => boot[:,1])
```

Here are the lower 95% confidence bounds for each method:

```output
3×4 DataFrame
 Row │ asym       rob        jack       boot      
     │ Float64    Float64    Float64    Float64   
─────┼────────────────────────────────────────────
   1 │  0.872755   0.954257   0.895928   0.912742
   2 │ -0.443074  -0.439597  -0.475232  -0.485941
   3 │  1.32749    1.32809    1.28391    1.32294
```


[^cfw]: Cheng SC, Fine JP, Wei LJ. Prediction of Cumulative Incidence Function under the Proportional Hazards Model. Biometrics. 1998;54:219–228.
