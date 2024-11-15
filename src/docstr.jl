####### module structs
DOC_ABSTRACTLSURVRESP = raw"""
AbstractLsurvResp

Abstract type representing a model response vector
"""

DOC_ABSTRACTLSURVPARMS = raw"""
AbstractLsurvParms

Abstract type representing a model predictors and coefficient parameters
"""

DOC_ABSTRACTPH = raw"""
Abstract type for proportional hazards models
"""

DOC_ABSTRACTNPSURV = raw"""
Abstract type for non-parametric survival models, including Kaplan-Meier, Aalen Johansen, and Cox-model based estimates of survival using an Aalen-Johansen-like estimator
"""

DOC_LSURVRESP = raw"""

 Outcome type for survival outcome subject to left truncation and right censoring. 

  Will not generally be needed by users

  Parameters
- `enter`: Time at observation start
- `exit`: Time at observation end
- `y`: event occurrence in observation
- `wts`: observation weights
- `eventtimes`: unique event times
- `origin`: origin on the time scale
- `id`: person level identifier (must be wrapped in ID() function)


```julia
 struct LSurvivalResp{
 E<:AbstractVector,
 X<:AbstractVector,
 Y<:AbstractVector,
 W<:AbstractVector,
 T<:Real,
 I<:AbstractLSurvivalID,
 } <: AbstractLSurvivalResp
 enter::E
 exit::X
 y::Y
 wts::W
 eventtimes::E
 origin::T
 id::Vector{I}
 end

```

```julia
 LSurvivalResp(
    enter::E,
    exit::X,
    y::Y,
    wts::W,
    id::Vector{I},
  ) where {
    E<:Vector,
    X<:Vector,
    Y<:Union{Vector{<:Real},BitVector},
    W<:Vector,
    I<:AbstractLSurvivalID,
}
```

```julia
 LSurvivalResp(
 enter::E,
 exit::X,
 y::Y,
 id::Vector{I},
 ) 

```

```julia
 LSurvivalResp(
  y::Vector{Y},
  wts::W,
  id::Vector{I},
  ) where {Y<:AbstractSurvTime,W<:Vector,I<:AbstractLSurvivalID}
```

```julia
 LSurvivalResp(
  enter::E,
  exit::X,
  y::Y,
  ) where {E<:Vector,X<:Vector,Y<:Union{Vector{<:Real},BitVector}}
```

```julia
 LSurvivalResp(exit::X, y::Y) where {X<:Vector,Y<:Vector}
```

  # Examples

```julia
  # no late entry
  LSurvivalResp([.5, .6], [1,0])

```

"""

DOC_LSURVCOMPRESP = raw"""

  Outcome type for competing risk survival outcomes subject to left truncation and right censoring (not generally needed for users)

  Parameters
- `enter` Time at observation start
- `exit` Time at observation end
- `y` event occurrence in observation
- `wts` observation weights
- `eventtimes` unique event times
- `origin` origin on the time scale
- `id` person level identifier (must be wrapped in ID() function)
- `eventtypes` vector of unique event types
- `eventmatrix` matrix of indicators on the observation level

  ## Signatures:

```julia
 struct LSurvivalCompResp{
 E<:AbstractVector,
 X<:AbstractVector,
 Y<:AbstractVector,
 W<:AbstractVector,
 T<:Real,
 I<:AbstractLSurvivalID,
 V<:AbstractVector,
 M<:AbstractMatrix,
 } <: AbstractLSurvivalResp
 enter::E
 exit::X
 y::Y
 wts::W
 eventtimes::X
 origin::T
 id::Vector{I}
 eventtypes::V
 eventmatrix::M
 end
```

```julia
 LSurvivalCompResp(
 enter::E,
 exit::X,
 y::Y,
 wts::W,
 id::Vector{I}
 )
```

```julia
 LSurvivalCompResp(
 enter::E,
 exit::X,
 y::Y,
 id::Vector{I}
 )
```

```julia
 LSurvivalCompResp(
 enter::E,
 exit::X,
 y::Y,
 wts::W,
 )
```

```julia
 LSurvivalCompResp(
 enter::E,
 exit::X,
 y::Y,
 )
```

```julia
 LSurvivalCompResp(
  exit::X,
  y::Y,
  ) where {X<:Vector,Y<:Union{Vector{<:Real},BitVector}}
```
"""

DOC_CONFINT = raw"""
```julia
using LSurvival
 dat1= (
   time = [1,1,6,6,8,9],
   status = [1,0,1,1,0,1],
   x = [1,1,1,0,0,0]
 )

 ft = coxph(@formula(Surv(time, status) ~ x),dat1, keepx=true)
 # model-based variance
 confint(ft)

 # robust variance
 confint(ft, type="robust")
```

 # for cluster confidence intervals

```julia
 dat1clust= (
   id = [1,2,3,3,4,4,5,5,6,6],
   enter = [0,0,0,1,0,1,0,1,0,1],
   exit = [1,1,1,6,1,6,1,8,1,9],
   status = [1,0,0,1,0,1,0,0,0,1],
   x = [1,1,1,1,0,0,0,0,0,0]
 )

 ft2 = coxph(@formula(Surv(enter, exit, status) ~ x),dat1clust, id=ID.(dat1clust.id), keepx=true)
 # model-based variance
 confint(ft2)

 # robust variance
 confint(ft2, type="robust")
```


"""

DOC_PHMODEL = raw"""
 PHModel: Mutable object type for proportional hazards regression (not generally needed for users)

  Parameters
- `R` Survival response
- `P`        # parameters
- `ties` String: "efron" or "breslow"
- `fit` Bool: logical for whether the model has been fitted
- `bh` AbstractMatrix: baseline hazard estimates

  ## Signatures

```julia
 mutable struct PHModel{G<:LSurvivalResp,L<:AbstractLSurvivalParms} <: AbstractPH
 R::G        # Survival response
 P::L        # parameters
 ties::String #"efron" or"breslow"
 fit::Bool
 bh::AbstractMatrix
 end

 PHModel(
 R::G,
 P::L,
 ties::String,
 fit::Bool,
 ) where {G<:LSurvivalResp,L<:AbstractLSurvivalParms}
 PHModel(R::G, P::L, ties::String) where {G<:LSurvivalResp,L<:AbstractLSurvivalParms}
 PHModel(R::G, P::L) where {G<:LSurvivalResp,L<:AbstractLSurvivalParms}
```
 Methods: fit, coef, confint, std_err, show

 # Example

```@example
 using LSurvival
 using Random
 import LSurvival: _stepcox!, dgm_comprisk

 z,x,t,d, event,wt = dgm_comprisk(MersenneTwister(1212), 100);
 enter = zeros(length(t));
 X = hcat(x,z);
 R = LSurvivalResp(enter, t, Int.(d), wt)
 P = PHParms(X)
 mf = PHModel(R,P)
  LSurvival._fit!(mf)
```
"""

DOC_PHSURV = raw"""
Mutable type for proportional hazards models (not generally needed by users)

PHSsurv: Object type for proportional hazards regression


surv::Vector{Float64}
risk::Matrix{Float64}
basehaz::Vector{Float64}
event::Vector{Float64}

- `fitlist`: vector of PHSurv objects (Cox model fits)
- `eventtypes`: vector of unique event types
- `times`: unique event times
- `surv`: Overall survival at each time
- `risk`: Cause-specific risk  at each time (1 for each outcome type)
- `basehaz`: baseline hazard for a specific event type
- `event`: value of event type that occurred at each time

Methods: fit, show

```julia
mutable struct PHSurv{G<:Array{T} where {T<:PHModel}} <: AbstractNPSurv
fitlist::G        
eventtypes::AbstractVector
times::AbstractVector
surv::Vector{Float64}
risk::Matrix{Float64}
basehaz::Vector{Float64}
event::Vector{Float64}
end

PHSurv(fitlist::Array{T}, eventtypes) where {T<:PHModel}
PHSurv(fitlist::Array{T}) where {T<:PHModel}
```
"""

DOC_ID = raw"""
Type for identifying individuals in survival outcomes.


Used for the id argument in 
 - Outcome types: LSurvivalResp, LSurvivalCompResp 
 - Model types: PHModel, KMRisk, AJRisk

 Accepts any Number or String. There is no significance to having this particular struct, but it enables easier use of multiple dispatch.

```@example
[ID(i) for i in 1:10]
```
"""

DOC_STRATA = raw"""
Type for identifying individuals in survival outcomes.
Used for the strata argument in PHModel (not yet implemented)

Accepts any Number or String. There is no significance to having this particular struct, but it enables easier use of multiple dispatch.

```julia
[Strata(i) for i in 1:10]
```
"""

####### Primary methods


DOC_COXPH = raw"""
 Fit a generalized Cox proportional hazards model to data.

 Alias for`fit(PHModel, ..., <keyword arguments>)`.

Also see ?fit section on "Fit method for AbstractPH objects"


 ## Signatures

 ## using matrix/vector interface

```julia
coxph(X::AbstractMatrix, enter::AbstractVector, exit::AbstractVector, y::AbstractVector; <keyword arguments>)
coxph(X::AbstractMatrix, exit::AbstractVector, y::AbstractVector; <keyword arguments>) # enter assumed to be 0
coxph(X::AbstractMatrix, exit::AbstractVector; <keyword arguments>) # enter assumed to be 0, y assumed to be 1
```
Parameters
- `X`: a design matrix (matrix of predictors)
- `enter`: Time at observation start
- `exit`: Time at observation end
- `y`: event occurrence in observation

Keyword parameters
- `ties` method for ties ("efron" or "breslow")
- `wts` vector of observation weights
- `keepx` (boolean) keep the design matrix after fitting the model? (default `true`, `false` can be used to save space)
- `keepy` (boolean) keep the outcome vector after fitting the model? (default `true`, `false` can be used to save space)

See [https://en.wikipedia.org/wiki/Proportional_hazards_model#Likelihood_when_there_exist_tied_times](for the log-partial liklihoods given for Efron's and Breslow's method)


## using StatsAPI `@formula` interface

```julia
coxph(f::Formula, dat; <keyword arguments>)
```
Parameters
- `f`: a `@formula` object
- `dat`: a Tables.jl compatible table

Keyword parameters
- `ties` method for ties ("efron" or "breslow")
- `wts` vector of observation weights
- `keepx` (boolean) keep the design matrix after fitting the model? (default `true`, `false` can be used to save space)
- `keepy` (boolean) keep the outcome vector after fitting the model? (default `true`, `false` can be used to save space)
- `contrasts` an optional Dict used to process the columns in `dat` (CF: See the contrasts argument in GLM.glm, and https://juliastats.org/StatsModels.jl/stable/contrasts/)

 # Example

```julia
using LSurvival
using Random
z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 200);
enter = zeros(length(t));
X = hcat(x,z);
tab = (enter=enter, t=t, d=d, x=x[:],z=z[:])

# three identical ways to fit the model
m1 = fit(PHModel, X, enter, t, d, ties="breslow")
m2 = coxph(X, enter, t, d, ties="breslow", keepx=false, keepy=false)                      
m3 = coxph(@formula(Surv(enter, t, d)~x+z), tab, ties="breslow", keepx=true, keepy=true)
coeftable(m)
stderror(m3)
stderror(m3, type="robust")
confint(m3, type="robust")
```

 # cluster robust standard errors using the `id` keyword argument

```julia
 dat1clust= (
   id = [1,2,3,3,4,4,5,5,6,6],
   enter = [0,0,0,1,0,1,0,1,0,1],
   exit = [1,1,1,6,1,6,1,8,1,9],
   status = [1,0,0,1,0,1,0,0,0,1],
   x = [1,1,1,1,0,0,0,0,0,0]
 )

 ft2 = coxph(@formula(Surv(enter, exit, status) ~ x),dat1clust, id=ID.(dat1clust.id), keepx=true)
 # model-based variance
 stderror(ft2)

 # robust variance
 stderror(ft2, type="robust")

 confint(ft2, type="robust")

 # once robust SE is calculated, coefficient table uses the robust SE for confidence intervals and test statistics
 ft2
```
"""


DOC_FIT_PHSURV = raw"""
Competing risk models:

Calculate survival curve and cumulative incidence (risk) function, get a set of Cox models (PHModel objects) that are exhaustive for the outcome types

```julia
fit(::Type{M},
fitlist::AbstractVector{<:T},
;
fitargs...) where {M<:PHSurv, T <: PHModel}
```
OR 

```julia
risk_from_coxphmodels(fitlist::Array{T}, args...; kwargs...) where T <: PHModel
```

fit for PHSurv objects

```@example
using LSurvival
using Random
 z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
 enter = zeros(length(t));
 X = hcat(x,rand(length(x)));

 ft1 = coxph(X, enter, t, (event .== 1), ties="breslow");
 nft2 = findall(event .!= 1)
 ft2 = coxph(X[nft2,:], enter[nft2], t[nft2], (event .== 2)[nft2], ties="breslow");
 fitlist = [ft1, ft2]

# Risk at x=0, z=0 (referent values)
# these are equivalent
 res = fit(PHSurv, [ft1, ft2])
 res2 = risk_from_coxphmodels([ft1, ft2])

# Risk at x=1, z=0.5
 res3 = risk_from_coxphmodels([ft1, ft2], pred_profile=[1.0, 0.5])

 # method using Aalen-Johansen analog (default)
 res3 = risk_from_coxphmodels([ft1, ft2], pred_profile=[1.0, 0.5])
 # method using Cheng, Fine and Wei's cumulative hazards based approach
 res3b = risk_from_coxphmodels([ft1, ft2], pred_profile=[1.0, 0.5], method="cheng-fine-wei")
```
Cheng SC, Fine JP, Wei LJ. Prediction of Cumulative Incidence Function under the Proportional Hazards Model. Biometrics. 1998;54:219–228.
"""

DOC_E_YEARSOFLIFELOST = raw"""
# Deprecated function

Expected number of years of life lost due to cause k

```julia
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

``` 
"""

####### generic methods

DOC_FIT_ABSTRACPH = raw"""
  Fit method for AbstractPH objects (Cox models)

  Keyword arguments (used here, and passed on to internal structs)

- `ties` "breslow" or "efron" (default)
- `wts` observation weights
- `ties` "breslow" or "efron" (default)
- `offset` not currently used at all
- `fitargs` arguments passed to other structs, which include
  - `id` cluster or individual level ID (defaults to a unique value for each row of data) see note below on ID
  - `contrasts` StatsModel style contrasts (dicts) that can be used for variable transformations/indicator variable creation (e.g. https://juliastats.org/StatsModels.jl/stable/contrasts/)
- Arguments passed onto fitting routine:
  - `eps` (default: Float64 = 1e-9) tolerance for declaring convergence. Model is determined to be converged when relative change in log-partial likelihood is < `eps` .
  - `getbasehaz` (default: true): estimate baseline hazard
  - `start` (default: nothing) nothing, or vector of floats corresponding to initial values for parameters. Note that this defaults to a vector of zeros when set to nothing, and setting to other values invalidates some of the test statistics reported by default with `coxph.`
  - `keepx` (default: true) logical. Keep design matrix in AbstractPH object output (set to false for slight computational gain).
  - `keepy` (default: true)logical.  Keep outcome in AbstractPH object output (set to false for slight computational gain).
  - `bootstrap_sample` (default: false) Fit the model to a bootstrap sample of the data (not generally used by end-users, but provides some convenience toward bootstrap variance estimation).
  - `bootstrap_rng` (default: Random.MersenneTwister()) Random number seed used when drawing a bootstrap sample of the data (not generally used by end-users, but provides some convenience toward bootstrap variance estimation).

  ## Signatures
```julia
  fit(::Type{M},
  X::AbstractMatrix,#{<:FP},
  enter::AbstractVector{<:Real},
  exit::AbstractVector{<:Real},
  y::Union{AbstractVector{<:Real},BitVector}
  ;
  ties ="breslow",
  wts::AbstractVector{<:Real}      = similar(y, 0),
  offset::AbstractVector{<:Real}   = similar(y, 0),
  fitargs...) where {M<:AbstractPH}
```

```
 coxph(f::FormulaTerm, data; kwargs...)
```

```
  coxph(X, enter, exit, y, args...; kwargs...)
```

```julia
   using LSurvival, Random
   z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
   enter = zeros(length(t));
   X = hcat(x,rand(length(x)));
    m = fit(PHModel, X, enter, t, d, ties="efron")
   m2 = fit(PHModel, X, enter, t, d, ties="breslow")
   coeftable(m)
```

 ## Note on use of `id` keyword

  `id` is not needed in person-period structure data for standard estimates or confidence intervals

```@example
  using Random, LSurvival
     id, int, outt, dat =
         LSurvival.dgm(MersenneTwister(123123), 100, 100; afun = LSurvival.int_0)
     data = (
             int = int,
             outt = outt,
             d = dat[:,4] .== 1,
             x = dat[:,1],
             z = dat[:,2]
     )

     f = @formula(Surv(int, outt,d)~x+z)
     coxph(f, data)
```

 # BUT, you must specify `id` to get appropriate robust variance and some other statistics.

 Here is an example where the same data are presented in two different ways, which should yield identical statistics when used in Cox model.
```@example
 dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
  )
  ft = coxph(@formula(Surv(time,status)~x),dat1)
  bic(ft)
  nobs(ft)
  dof_residual(ft)
  # lrtest is another one

  stderror(ft)                     # model based
  stderror(ft, type="robust")   # robust standard error, based on dfbeta residuals
  ft

  # now using "clustered" data with multiple observations per individual
 dat1clust= (
     id = [1,2,3,3,4,4,5,5,6,6],
     enter = [0,0,0,1,0,1,0,1,0,1],
     exit = [1,1,1,6,1,6,1,8,1,9],
     status = [1,0,0,1,0,1,0,0,0,1],
     x = [1,1,1,1,0,0,0,0,0,0]
 )
 
 # use the `id` parameter with the ID struct
 ft2 = coxph(@formula(Surv(enter, exit, status) ~ x),dat1clust, id=ID.(dat1clust.id))
 bic(ft2)                       # CORRECT        
 nobs(ft2)                      # CORRECT
 dof_residual(ft2)              # CORRECT
  
 stderror(ft2)                  # model based (CORRECT)
 stderror(ft2, type="robust")   # robust standard error, based on `id` level dfbeta residuals (CORRECT)
 # once robust SE is calculated, coefficient table uses the robust SE for confidence intervals and test statistics
 ft2   # CORRECT (compare to `ft` object)
```
 
 ## NOTE THE FOLLOWING IS INCORRECT because the `id` keyword is omitted

```@example
 ft2w = coxph(@formula(Surv(enter, exit, status) ~ x),dat1clust)
 bic(ft2w)                          # INCORRECT 
 nobs(ft2w)                         # INCORRECT
 dof_residual(ft2w)                 # INCORRECT

 stderror(ft2w)                     # model based (CORRECT)
 stderror(ft2w, type="robust")      # robust variance (INCORRECT)
 
 ft2w # the coefficient table now shows incorrect confidence intervals and test statistics
  
```
"""

DOC_FIT_KMSURV = raw"""
Kaplan-Meier estimator for cumulative conditional risk

 ## Signatures

```julia
StatsBase.fit!(m::T; kwargs...) where {T<:AbstractNPSurv}

kaplan_meier(enter::AbstractVector, exit::AbstractVector, y::AbstractVector,
   ; kwargs...)
```

## Keyword arguments
- wts::Vector{<:Real} = similar(enter, 0); vector of case weights (or zero length vector) for each observation
- id::Vector{<:AbstractLSurvivalID} = [ID(i) for i in eachindex(y)]; Vector of AbstractSurvID objects denoting observations that form a single unit (used in bootstrap and jackknife methods)
- atol = 0.00000001; absolute tolerance for defining tied event times
- censval = 0;  value of the outcome to be considered a censored event
- keepy = true; keep the outcome vector after fitting (may save memory with large datasets)
- eps = 0.00000001; deprecated (replaced by atol)

```@example
using LSurvival
using Random
z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
enter = zeros(length(t));
m = fit(KMSurv, enter, t, d)
mw = fit(KMSurv, enter, t, d, wts=wt)
```
or, equivalently:

```julia
kaplan_meier(enter, t, d, wts=wt)
```
"""

DOC_FIT_AJSURV = raw"""
Aalen-Johansen estimator for cumulative cause-specific risk (in the presence of competing events)

## Signatures

```julia
 StatsBase.fit!(m::T; kwargs...) where {T<:AbstractNPSurv}

 aalen_johansen(enter::AbstractVector, exit::AbstractVector, y::AbstractVector,
   ; kwargs...)

```

## Keyword arguments
- `wts::Vector{<:Real} = similar(enter, 0)`; vector of case weights (or zero length vector) for each observation
- `id::Vector{<:AbstractLSurvivalID} = [ID(i) for i in eachindex(y)]`; Vector of AbstractSurvID objects denoting observations that form a single unit (used in bootstrap and jackknife methods)
- `atol = 0.00000001`; absolute tolerance for defining tied event times
- `keepy = true`; keep the outcome vector after fitting (may save memory with large datasets)
- `eps = 0.00000001`; deprecated (replaced by atol)


```@example
using LSurvival
using Random
z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
enter = zeros(length(t));
   # event variable is coded 0[referent],1,2
m = fit(AJSurv, enter, t, event)
mw = fit(AJSurv, enter, t, event, wts=wt)
```
or, equivalently:

```julia
aalen_johansen(enter, t, event, wts=wt)
```
"""

DOC_VARIANCE_KMSURV = raw"""
Greenwood's formula for variance and confidence intervals of a Kaplan-Meier survival curve


 ## Signatures:

```julia
StatsBase.stderror(m::KMSurv)

StatsBase.confint(m:KMSurv; level=0.95, method="normal")
```

## Keyword arguments

`method`:
  - "normal" normality-based confidence intervals
  - "lognlog" log(-log(S(t))) based confidence intervals


```@example
using LSurvival
using Random
z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
enter = zeros(length(t));
m = fit(KMSurv, enter, t, d)
mw = fit(KMSurv, enter, t, d, wts=wt)
stderror(m)
confint(m, method="normal")
confint(m, method="lognlog") # log-log transformation
```
"""

DOC_VARIANCE_AJSURV = raw"""
Greenwood's formula for variance and confidence intervals of a Aalen-Johansen risk function

## Signatures:

```julia
StatsBase.stderror(m::AJSurv)

StatsBase.confint(m:AJSurv; level=0.95, method="normal")
```

## Keyword arguments
- method
  - "normal" normality-based confidence intervals
  - "lognlog" log(-log(S(t))) based confidence intervals


```@example
res = z, x, outt, d, event, wts = LSurvival.dgm_comprisk(MersenneTwister(123123), 100)
int = zeros(length(d)) # no late entry
m = fit(AJSurv, int, outt, event)
stderror(m)
confint(m, level=0.95)
```
"""




####### non-user functions

DOC_LGH_BRESLOW = raw"""
 Update the partial likelihood, gradient and Hessian values from a Cox model fit (used during fitting, not generally useful for users).

 Uses Breslow's partial likelihood.

 Updates over all observations

 ## Signature 

```julia
lgh_breslow!(m::M, j, caseidx, risksetidx) where {M<:AbstractPH}
```

"""

DOC_LGH_EFRON = raw"""
 Update the partial likelihood, gradient and Hessian values from a Cox model fit (used during fitting, not generally useful for users).

 Uses Efron's partial likelihood.

 Updates over all observations

 ## Signature 

```julia
lgh_efron!(m::M, j, caseidx, risksetidx) where {M<:AbstractPH}
```
"""

DOC_LGH = raw"""
 Update the partial likelihood, gradient and Hessian values from a Cox model fit (used during fitting, not generally useful for users).

 Uses Breslow's or Efron's partial likelihood.

 Updates over all a single observation. This is just a simple wrapper that calls `lgh_breslow!` or `lgh_efron!`

 ## Signature 

```julia
lgh!(m::M, j, caseidx, risksetidx) where {M<:AbstractPH}
```
"""


DOC__UPDATE_PHPARMS = """
 Update the partial likelihood, gradient and Hessian values from a Cox model fit (used during fitting, not generally useful for users).

 Uses Breslow's or Efron's partial likelihood.

 Updates over all observations

## Signature 

```julia
_update_PHParms!(
 m::M,
 # big indexes
 ne::I,
 caseidxs::Vector{Vector{T}},
 risksetidxs::Vector{Vector{T}},
 ) where {M<:AbstractPH,I<:Int,T<:Int}
```
"""

DOC_FIT_PHSURV = """
  Survival curve estimation using multiple cox models


  ## Signatures

```julia
  risk_from_coxphmodels(fitlist::Vector{T}, args...; kwargs...) where {T<:PHModel}

  fit(::Type{M}, fitlist::Vector{T}, ; fitargs...) where {M<:PHSurv,T<:PHModel}
```

  ## Optional keywords
- `coef_vectors` = nothing(default) or vector of coefficient vectors from the cox models [will default to the coefficients from fitlist models]
- `pred_profile` = nothing(default) or vector of specific predictor values of the same length as the coef_vectors[1]

```@example
 using LSurvival
 using Random
 # event variable is coded 0[referent],1,2
 z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
 enter = zeros(length(t));

 ft1 = coxph(hcat(x,z), enter, t, (event .== 1))
 nidx = findall(event .!= 1)
 ft2 = coxph(hcat(x,z)[nidx,:], enter[nidx], t[nidx], (event[nidx] .== 2))

 # risk at referent levels of `x` and `z`
 risk_from_coxphmodels([ft1,ft2])

 # risk at average levels of `x` and `z`
 mnx = sum(x)/length(x)
 mnz = sum(z)/length(z)
 # equivalent
 fit(PHSurv, [ft1,ft2], pred_profile=[mnx,mnz])
 risk_from_coxphmodels([ft1,ft2], pred_profile=[mnx,mnz])
```
"""


####### data generation functions
DOC_DGM = raw"""
Generating discrete survival data without competing risks

## Signatures
```
dgm(rng::MersenneTwister, n::Int, maxT:Int; afun = int_0, yfun = yprob, lfun = lprob)

dgm(n::Int, maxT::Int; kwargs...)
```

Usage: dgm(rng, n, maxT;afun=int_0, yfun=yprob, lfun=lprob)
dgm(n, maxT;afun=int_0, yfun=yprob, lfun=lprob)

Where afun, yfun, and lfun are all functions that take arguments v,l,a and output time-specific values of a, y, and l respectively
Example:

```julia

expit(mu) =  inv(1.0+exp(-mu))

function aprob(v,l,a)
expit(-1.0 + 3*v + 2*l)
end
  
function lprob(v,l,a)
expit(-3 + 2*v + 0*l + 0*a)
end
  
function yprob(v,l,a)
expit(-3 + 2*v + 0*l + 2*a)
end
  # 10 individuals followed for up to 5 times
LSurvival.dgm(10, 5;afun=aprob, yfun=yprob, lfun=lprob)
```

"""

DOC_DGM_COMPRISK = raw"""
Generating continuous survival data with competing risks

## Signatures

```
dgm_comprisk(rng::MersenneTwister, n::Int)

dgm_comprisk(n::Int)
```

        - rng = random number generator    
        - n = sample size

Example:

```julia
using LSurvival
# 100 individuals with two competing events
z,x,t,d,event,weights = LSurvival.dgm_comprisk(100)
    

```
"""

DOC_BOOTSTRAP_PHMODEL = raw"""
Bootstrapping coefficients of a proportional hazards model

## Signatures

```
# single bootstrap draw, keeping the entire object
bootstrap(rng::MersenneTwister, m::PHModel)
bootstrap(m::PHModel)
# muliple bootstrap draws, keeping only coefficient estimates
bootstrap(rng::MersenneTwister, m::PHModel, iter::Int; kwargs...)
bootstrap(m::PHModel, iter::Int; kwargs...)
```

Returns:
- If using `bootstrap(m)`: a single bootstrap draw
- If using `bootstrap(m, 10)` (e.g.): 10 bootstrap draws of the cumulative cause-specific risks at the end of follow up




```julia
using LSurvival, Random

id, int, outt, data =
LSurvival.dgm(MersenneTwister(1212), 500, 5; afun = LSurvival.int_0)

d, X = data[:, 4], data[:, 1:3]
weights = rand(length(d))

# survival outcome:
R = LSurvivalResp(int, outt, d, ID.(id))    # specification with ID only
P = PHParms(X)

Mod = PHModel(R, P)
LSurvival._fit!(Mod, start=Mod.P._B, keepx=true, keepy=true)


# careful propogation of bootstrap sampling
idx, R2 = bootstrap(R)
P2 = bootstrap(idx, P)
Modb = PHModel(R2, P2)
LSurvival._fit!(Mod, start=Mod.P._B, keepx=true, keepy=true)

# convenience function for bootstrapping a model
Modc = bootstrap(Mod)
LSurvival._fit!(Modc, start=Modc.P._B);
Modc
Modc.P.X == nothing
Modc.R == nothing

```

Bootstrap Cox model coefficients

```
LSurvival._fit!(mb, keepx=true, keepy=true, start=[0.0, 0.0])
```

```@example
using LSurvival, Random
res = z, x, outt, d, event, wts = LSurvival.dgm_comprisk(MersenneTwister(123123), 200)
int = zeros(length(d)) # no late entry
X = hcat(z, x)

mainfit = fit(PHModel, X, int, outt, d .* (event .== 1), keepx=true, keepy=true)

function stddev_finite(x)
 n = length(x)
 mnx = sum(x)/n
 ret = sum((x .- mnx) .^ 2)
 ret /= n-1
 sqrt(ret)
end

# bootstrap standard error versus asymptotic
mb = bootstrap(MersenneTwister(123123), mainfit, 200)
## bootstrap standard error
[stddev_finite(mb[:,i]) for i in 1:2]
## asymptotic standard error
stderror(mainfit)

```
"""

DOC_BOOTSTRAP_KMSURV = raw"""
Bootstrap methods for Kaplan-Meier survival curve estimator

 ## Signatures

```
 # single bootstrap draw, keeping the entire object
 bootstrap(rng::MersenneTwister, m::KMSurv)
 bootstrap(m::KMSurv)

 # muliple bootstrap draws, keeping only coefficient estimates
 bootstrap(rng::MersenneTwister, m::KMSurv, iter::Int; kwargs...)
 bootstrap(m::KMSurv, iter::Int; kwargs...)
```

 Returns:
  - If using `bootstrap(m)`: a single bootstrap draw
  - If using `bootstrap(m, 10)` (e.g.): 10 bootstrap draws of the survival probability at the end of follow up
  
  

```@example
using LSurvival
using Random

id, int, outt, data =
LSurvival.dgm(MersenneTwister(1212), 20, 5; afun = LSurvival.int_0)

d, X = data[:, 4], data[:, 1:3]
wts = rand(length(d))

km1 = kaplan_meier(int, outt, d, id=ID.(id), wts=wts)
km2 = bootstrap(km1, keepy=false)
km3 = bootstrap(km1, 10, keepy=false)
km1

km1.R
km2.R

```
"""

DOC_BOOTSTRAP_AJSURV = raw"""
Bootstrap methods for Aalen-Johansen cumulative risk estimator
 
 ## Signatures

```
 # single bootstrap draw, keeping the entire object
 bootstrap(rng::MersenneTwister, m::AJSurv)
 bootstrap(m::AJSurv)

 # muliple bootstrap draws, keeping only coefficient estimates
 bootstrap(rng::MersenneTwister, m::AJSurv, iter::Int; kwargs...)
 bootstrap(m::AJSurv, iter::Int; kwargs...)
```

Returns:
- If using `bootstrap(m)`: a single bootstrap draw
- If using `bootstrap(m, 10)` (e.g.): 10 bootstrap draws of the cumulative cause-specific risks at the end of follow up


```@example
using LSurvival
using Random

z, x, t, d, event, wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 100)
id = 1:length(x)
enter = zeros(length(t))

aj1 = aalen_johansen(enter, t, event, id=ID.(id), wts=wt)
aj2 = bootstrap(aj1, keepy=false);
ajboot = bootstrap(aj1, 10, keepy=false);
aj1


aj1.R
aj2.R

```
"""

DOC_BOOTSTRAP_SURVRESP = raw"""
Bootstrapping sampling of a survival response


```
id, int, outt, data =
LSurvival.dgm(MersenneTwister(1212), 20, 5; afun = LSurvival.int_0)

d, X = data[:, 4], data[:, 1:3]
weights = rand(length(d))

# survival outcome:
R = LSurvivalResp(int, outt, d, ID.(id))    # specification with ID only
```
"""


DOC_BOOTSTRAP_SURVCOMPRESP = raw"""
Bootstrapping sampling of a competing risk survival response

## Signatures

```
bootstrap(rng::MersenneTwister, R::T) where {T<:LSurvivalCompResp}
bootstrap(R::T) where {T<:LSurvivalCompResp}
```

```
z,x,t,d,event,weights =
LSurvival.dgm_comprisk(MersenneTwister(1212), 300)
enter = zeros(length(event))

# survival outcome:
R = LSurvivalCompResp(enter, t, event, weights, ID.(collect(1:length(t))))    # specification with ID only
bootstrap(R) # note that entire observations/clusters identified by id are kept
```
"""

DOC_BOOTSTRAP_PHPARMS = raw"""
Bootstrap sampling of a proportional hazards predictor object

```
using LSurvival, Random

id, int, outt, data =
LSurvival.dgm(MersenneTwister(1212), 20, 5; afun = LSurvival.int_0)

d, X = data[:, 4], data[:, 1:3]
weights = rand(length(d))

# survival outcome:
R = LSurvivalResp(int, outt, d, ID.(id))    # specification with ID only
P = PHParms(X)
idx, R2 = bootstrap(R)
P2 = bootstrap(idx, P)

Mod = PHModel(R2, P2)
LSurvival._fit!(Mod, start=Mod.P._B)

```

"""


DOC_JACKKNIFE = raw"""
Obtain jackknife (leave-one-out) estimates from a Cox model by refitting the model n times

```@example
using LSurvival, Random, StatsBase
id, int, outt, data =
LSurvival.dgm(MersenneTwister(112), 100, 10; afun = LSurvival.int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))
wt ./= (sum(wt) / length(wt))
m = coxph(X,int, outt,d, wts=wt, id=ID.(id))

jk = jackknife(m);
bs = bootstrap(MersenneTwister(12321), m, 1000);
N = nobs(m)
#comparing estimate with jackknife estimate with bootstrap mean
hcat(coef(m), mean(jk, dims=1)[1,:], mean(bs, dims=1)[1,:])
semb = stderror(m)
sebs = std(bs, dims=1)
sero = stderror(m, type="robust")
sejk = stderror(m, type="jackknife")
sejk_manual = std(jk, dims=1, corrected=false) .* sqrt(N-1)

sqrt.(diag(LSurvival.jackknife_vcov(m)))

hcat(semb, sebs[1,:], sejk, sejk_manual[1,:], sero)

dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])
dat1clust = (
  id = [1, 2, 3, 3, 4, 4, 5, 5, 6, 6],
  enter = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
  exit = [1, 1, 1, 6, 1, 6, 1, 8, 1, 9],
  status = [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
  x = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
)

m = coxph(@formula(Surv(time, status)~x),dat1)
mc = coxph(@formula(Surv(enter, exit, status)~x),dat1clust, id=ID.(dat1clust.id))
jk = jackknife(m);
jkc = jackknife(mc);
bs = bootstrap(mc, 100);
std(bs[:,1])
stderror(m, type="jackknife")
stderror(mc, type="jackknife")
@assert jk == jkc
```
"""


DOC_RESIDUALS = raw"""
  ####################################################################
  
  Cox proportional hazards model residuals:

  Signature

```julia
  residuals(m::M; type = "martingale") where {M<:PHModel}
```
  where type is one of 
- `martingale`
- `schoenfeld`
- `score`
- `dfbeta`
- `jackknife`
- `dfbetas` (scaled dfbeta)
- `scaled_schoenfeld` or `schoenfelds`  (scaled Schoenfeld)

  Residuals from the residuals function are designed to exactly emulate those from the `survival` package in R. Currently, they are validated for single observation data (e.g. one data row per individual).

  ####################################################################

  ## Martingale residuals: Observed versus expected
  
```@example
  # example from https://cran.r-project.org/web/packages/survival/vignettes/validate.pdf
  # by Terry Therneau

  dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
  )

  # Nelson-Aalen type estimator for Breslow partial likelihood
  ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow")
  residuals(ft, type="martingale")
```

```@example
  dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
  )

  # Fleming-Harrington type estimator for Efron partial likelihood
  ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="efron")
  residuals(ft, type="martingale")

```
  ####################################################################

  ## Score residuals: Per observation contribution to score function 

```julia
  using LSurvival
  dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
  )
  ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow")
  S = residuals(ft, type="score")[:]
  ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="efron", maxiter=0)
  S = residuals(ft, type="score")[:]
```

  ####################################################################

  ## Schoenfeld residuals: Per time contribution to score function 

```julia
  using LSurvival
  dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
  )
  ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow", maxiter=0)


  X = ft.P.X
  M = residuals(ft, type="martingale")
  S = residuals(ft, type="schoenfeld")[:]
  Ss = residuals(ft, type="scaled_schoenfeld")[:]
```

  ####################################################################

  ## dfbeta residuals: influence of individual observations on each parameter

```@example
  using LSurvival
  dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
  )

  ft = coxph(@formula(Surv(time,status)~x),dat1, ties="breslow")
  residuals(ft, type="dfbeta")

  # can also calculate from score residuals and Hessian matrix
  L = residuals(ft, type="score") # n X p
  H = ft.P._hess   # p X p
  dfbeta = L*inv(H)
  robVar = dfbeta'dfbeta
  sqrt(robVar)

```

  #  using the `id` keyword argument
  #  see help for LSurvival.vcov for what happens when `id` keyword is not used

```@example
  dat1clust= (
    id = [1,2,3,3,4,4,5,5,6,6],
    enter = [0,0,0,1,0,1,0,1,0,1],
    exit = [1,1,1,6,1,6,1,8,1,9],
    status = [1,0,0,1,0,1,0,0,0,1],
    x = [1,1,1,1,0,0,0,0,0,0]
  )

  ft2 = coxph(@formula(Surv(enter, exit, status) ~ x),dat1clust, id=ID.(dat1clust.id), ties="breslow")

  # note these are still on the observation level (not the id level)! 
  residuals(ft2, type="dfbeta")

  # getting id level dfbeta residuals
  dfbeta = residuals(ft2, type="dfbeta")
  id = values(ft2.R.id)
  D = reduce(vcat, [sum(dfbeta[findall(id .== i),:], dims=1) for i in unique(id)])
  D'D
  vcov(ft, type="robust")
  vcov(ft2, type="robust")
```

  ####################################################################

  ## jackknife residuals: influence of individual observations on each parameter using leave-one-out estimates
 note there are other definitions of jackknife residuals
 See Chapter 7.1 of "Extending the Cox Model" by Therneau and Grambsch for an example of the type of jackknife residuals used here

  Jackknife residuals $r_i$ for $i \in 1:n$ are given as the difference between the maximum partial likelihood estimate and the jackknife estimates for each observation

$$r_i = \hat\beta - \hat\beta_{(-i)}$$

  where $\beta_{(-i)}$ is the maximum partial likelihood estimate of the log-hazard ratio vector obtained from a dataset in which observations belonging to individual `i` are removed

```@example
  using LSurvival
  dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
  )

  ft = coxph(@formula(Surv(time,status)~x),dat1, ties="breslow")
  #jackknife(ft)
  residuals(ft, type="jackknife")


```
"""

DOC_VCOV = raw"""
  Covariance matrix for Cox proportional hazards models  

  Keyword arguments
- `type` nothing or "robust": determines whether model based or robust (dfbeta based) variance is returned.

  See ?residuals for info on `dfbeta` residuals

```@example
using LSurvival
dat1 = (
  time = [1,1,6,6,8,9],
  status = [1,0,1,1,0,1],
  x = [1,1,1,0,0,0]
)
ft = coxph(@formula(Surv(time,status)~x),dat1, id=ID.(collect(1:6)))

vcov(ft)                   # model based
vcov(ft, type="robust")    # robust variance, based on dfbeta residuals
# once robust SE is calculated, coefficient table uses the robust SE for confidence intervals and test statistics
ft
```

# cluster robust standard errors using the `id` keyword argument

```@example
dat1clust= (
  id = [1,2,3,3,4,4,5,5,6,6],
  enter = [0,0,0,1,0,1,0,1,0,1],
  exit = [1,1,1,6,1,6,1,8,1,9],
  status = [1,0,0,1,0,1,0,0,0,1],
  x = [1,1,1,1,0,0,0,0,0,0]
)

ft2 = coxph(@formula(Surv(enter, exit, status) ~ x),dat1clust, id=ID.(dat1clust.id))

vcov(ft2)                     # model based
vcov(ft2, type="robust")       # robust variance, based on dfbeta residuals
stderror(ft2, type="robust")   # robust variance, based on dfbeta residuals
confint(ft2, type="robust")    # robust variance, based on dfbeta residuals
nobs(ft2)                     # id argument yields correct value of number of independent observations
# once robust SE is calculated, coefficient table uses the robust SE for confidence intervals and test statistics
ft2 
```

## NOTE THE FOLLOWING IS INCORRECT because the `id` keyword is omitted

```@example
ft2w = coxph(@formula(Surv(enter, exit, status) ~ x),dat1clust)

vcov(ft2w)                   # model based (CORRECT)
vcov(ft2w, type="robust")    # robust variance (INCORRECT)
nobs(ft2w)

ft2w
```
"""


######## DEPRECATED FUNCTIONS ###############

"""
*Deprecated function*

  Estimate parameters of an extended Cox model

  Using: Newton raphson algorithm with modified/adaptive step sizes

  Keyword inputs:
  method="efron", 
  inits=nothing , # initial parameter values, set to zero if this is nothing
  tol=10e-9,      #  convergence tolerance based on log likelihod: likrat = abs(lastLL/_LL[1]), absdiff = abs(lastLL-_LL[1]), reldiff = max(likrat, inv(likrat)) -1.0
  maxiter=500    # maximum number of iterations for Newton Raphson algorithm (set to zero to calculate likelihood, gradient, Hessian at the initial parameter values)

  Outputs:
  beta: coefficients 
  ll: log partial likelihood history (all iterations), with final value being the (log) maximum partial likelihood (log-MPL)
  g: gradient vector (first derivative of log partial likelihood) at log-MPL
  h: hessian matrix (second derivative of log partial likelihood) at log-MPL
  basehaz: Matrix: baseline hazard at referent level of all covariates, weighted risk set size, weighted # of cases, time


  Examples: 
```julia-repl   
  using LSurvival
  # simulating discrete survival data for 20 individuals at 10 time points
  id, int, outt, data = LSurvival.dgm(20, 5;afun=LSurvival.int_0);
  
  d,X = data[:,4], data[:,1:3]
  
  # getting raw values of output
  args = (int, outt, d, X)
  beta, ll, g, h, basehaz = coxmodel(args..., method="efron")
  beta2, ll2, g2, h2, basehaz2 = coxmodel(args..., method="breslow")


  # nicer summary of results
  args = (int, outt, d, X)
  res = coxmodel(args..., method="efron");
  coxsum = cox_summary(res, alpha=0.05, verbose=true);
    
```
"""

"""
*Deprecated function*
  Estimating cumulative incidence from two or more cause-specific Cox models
  
  z,x,outt,d,event,weights = LSurvival.dgm_comprisk(120)
  X = hcat(z,x)
  int = zeros(120)
  d1  = d .* Int.(event.== 1)
  d2  = d .* Int.(event.== 2)
  sum(d)/length(d)
  
  
  lnhr1, ll1, g1, h1, bh1 = coxmodel(int, outt, d1, X, method="efron");
  lnhr2, ll2, g2, h2, bh2 = coxmodel(int, outt, d2, X, method="efron");
  bhlist = [bh1, bh2]
  coeflist = [lnhr1, lnhr2]
  covarmat = sum(X, dims=1) ./ size(X,1)
  ci, surv = ci_from_coxmodels(bhlist;eventtypes=[1,2], coeflist=coeflist, covarmat=covarmat)
  ci, surv = ci_from_coxmodels(bhlist;eventtypes=[1,2])
"""

"""
*Deprecated function*

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

"""
*Deprecated function*

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


"""
*Deprecated function*

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

"""
*Deprecated function*

$DOC_E_YEARSOFLIFELOST
"""

