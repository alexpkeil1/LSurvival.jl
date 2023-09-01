####### module structs
DOC_ABSTRACTLSURVRESP ="""
 AbstractLsurvResp

 Abstract type representing a model response vector
"""

DOC_ABSTRACTLSURVPARMS ="""
 AbstractLsurvParms

 Abstract type representing a model predictors and coefficient parameters
"""

DOC_ABSTRACTPH ="""
 Abstract type for proportional hazards models
"""

DOC_ABSTRACTNPSURV ="""
 Abstract type for non-parametric survival models, including Kaplan-Meier, Aalen Johansen, and Cox-model based estimates of survival using an Aalen-Johansen-like estimator
"""

DOC_LSURVRESP ="""

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


```{julia}
 struct LSurvResp{
 E<:AbstractVector,
 X<:AbstractVector,
 Y<:AbstractVector,
 W<:AbstractVector,
 T<:Real,
 I<:AbstractLSurvID,
 } <: AbstractLSurvResp
 enter::E
 exit::X
 y::Y
 wts::W
 eventtimes::E
 origin::T
 id::Vector{I}
 end

```

```{julia}
 LSurvResp(
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
    I<:AbstractLSurvID,
}
```

```{julia}
 LSurvResp(
 enter::E,
 exit::X,
 y::Y,
 id::Vector{I},
 ) 

```

```{julia}
 LSurvResp(
  y::Vector{Y},
  wts::W,
  id::Vector{I},
) where {Y<:AbstractSurvTime,W<:Vector,I<:AbstractLSurvID}
```

```{julia}
 LSurvResp(
  enter::E,
  exit::X,
  y::Y,
) where {E<:Vector,X<:Vector,Y<:Union{Vector{<:Real},BitVector}}
```

```{julia}
 LSurvResp(exit::X, y::Y) where {X<:Vector,Y<:Vector}
```

# Examples

```{julia}
# no late entry
LSurvResp([.5, .6], [1,0])

```

"""

DOC_LSURVCOMPRESP ="""

Outcome type for competing risk survival outcomes subject to left truncation and right censoring (not generally needed for users)

Parameters
- `enter`: Time at observation start
- `exit`: Time at observation end
- `y`: event occurrence in observation
- `wts`: observation weights
- `eventtimes`: unique event times
- `origin`: origin on the time scale
- `id`: person level identifier (must be wrapped in ID() function)
- `eventtypes`: vector of unique event types
- `eventmatrix`: matrix of indicators on the observation level

# Signatures: 
```{julia}
 struct LSurvCompResp{
 E<:AbstractVector,
 X<:AbstractVector,
 Y<:AbstractVector,
 W<:AbstractVector,
 T<:Real,
 I<:AbstractLSurvID,
 V<:AbstractVector,
 M<:AbstractMatrix,
 } <: AbstractLSurvResp
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

```{julia}
 LSurvCompResp(
 enter::E,
 exit::X,
 y::Y,
 wts::W,
 id::Vector{I}
 )
```

```{julia}
 LSurvCompResp(
 enter::E,
 exit::X,
 y::Y,
 id::Vector{I}
 )
```

```{julia}
 LSurvCompResp(
 enter::E,
 exit::X,
 y::Y,
 wts::W,
 )
```

```{julia}
 LSurvCompResp(
 enter::E,
 exit::X,
 y::Y,
 )
```

```{julia}
 LSurvCompResp(
  exit::X,
  y::Y,
) where {X<:Vector,Y<:Union{Vector{<:Real},BitVector}}
```

"""

DOC_PHMODEL ="""
 PHModel: Mutable object type for proportional hazards regression (not generally needed for users)

Parameters
- `R`: Survival response
- `P`:        # parameters
- `ties`, String: "efron" or "breslow"
- `fit`, Bool: logical for whether the model has been fitted
- `bh`, AbstractMatrix: baseline hazard estimates

# Signatures

```{julia}
 mutable struct PHModel{G<:LSurvResp,L<:AbstractLSurvParms} <: AbstractPH
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
 ) where {G<:LSurvResp,L<:AbstractLSurvParms}
 PHModel(R::G, P::L, ties::String) where {G<:LSurvResp,L<:AbstractLSurvParms}
 PHModel(R::G, P::L) where {G<:LSurvResp,L<:AbstractLSurvParms}
```
 Methods: fit, coef, confint, std_err, show

 # Example

```{julia}
 using LSurvival
 using Random
 import LSurvival: _stepcox!, dgm_comprisk

 z,x,t,d, event,wt = dgm_comprisk(MersenneTwister(1212), 100);
 enter = zeros(length(t));
 X = hcat(x,z);
 R = LSurvResp(enter, t, Int.(d), wt)
 P = PHParms(X)
 mf = PHModel(R,P)
  LSurvival._fit!(mf)
```
"""

DOC_PHSURV ="""
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

```{julia}
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

DOC_ID ="""
 Type for identifying individuals in survival outcomes.


 Used for the id argument in 
  - Outcome types: LSurvResp, LSurvCompResp 
  - Model types: PHModel, KMRisk, AJRisk

Accepts any Number or String. There is no significance to having this particular struct, but it enables easier use of multiple dispatch.

```{julia}
 [ID(i) for i in 1:10]
```

"""

DOC_STRATA ="""
 Type for identifying individuals in survival outcomes.
 Used for the strata argument in PHModel (not yet implemented)

 Accepts any Number or String. There is no significance to having this particular struct, but it enables easier use of multiple dispatch.

```{julia}
 [Strata(i) for i in 1:10]
```

"""
####### Primary methods

DOC_COXPH ="""
Fit a generalized Cox proportional hazards model to data.

Alias for`fit(PHModel, ..., <keyword arguments>)`.


Signatures

```{julia}
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
 - ties: method for ties ("efron" or "breslow")
 
```{julia}
 coxph(f::Formula, dat; <keyword arguments>)
```
 Parameters
 - `f`: a `@formula` object
 - `dat`: a Tables.jl compatible table

 Keyword parameters
 - contrasts: an optional Dict used to process the columns in `dat` (CF: See the contrasts argument in GLM.glm)
 - ties: method for ties ("efron" or "breslow")

# Example

```{julia}
 using LSurvival
 using Random
 z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
 enter = zeros(length(t));
 X = hcat(x,rand(length(x)));
 m2 = fit(PHModel, X, enter, t, d, ties="breslow")
 coxph(X, enter, t, d, ties="breslow")
 coeftable(m)
```

"""


DOC_FIT_PHSURV ="""
 Competing risk models:

 Calculate survival curve and cumulative incidence (risk) function, get a set of Cox models (PHModel objects) that are exhaustive for the outcome types

```{julia}
 fit(::Type{M},
 fitlist::AbstractVector{<:T},
 ;
 fitargs...) where {M<:PHSurv, T <: PHModel}
```
 OR 

```{julia}
 risk_from_coxphmodels(fitlist::Array{T}, args...; kwargs...) where T <: PHModel

```


 fit for PHSurv objects

```{julia}
 using LSurvival
 using Random
 z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
 enter = zeros(length(t));
 X = hcat(x,rand(length(x)));
 #m2 = fit(PHModel, X, enter, t, d, ties="breslow")
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

```
"""

DOC_E_YEARSOFLIFELOST ="""
 # Deprecated function

 Expected number of years of life lost due to cause k

```{julia}
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

DOC_FIT_ABSTRACPH ="""
 Fit method for AbstractPH objects

```{julia}
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

```{julia}
 using LSurvival
 using Random
 z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
 enter = zeros(length(t));
 X = hcat(x,rand(length(x)));
  #R = LSurvResp(enter, t, Int.(d), wt)
  #P = PHParms(X,"efron")
  #mod = PHModel(R,P, true)
  #_fit!(mod)
 m = fit(PHModel, X, enter, t, d, ties="efron")
 m2 = fit(PHModel, X, enter, t, d, ties="breslow")
 coeftable(m)
```

```{julia}
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
"""

DOC_FIT_KMSURV ="""
 fit for KMSurv objects

  Signatures

```{julia}
 StatsBase.fit!(m::T; kwargs...) where {T<:AbstractNPSurv}

 kaplan_meier(enter::AbstractVector, exit::AbstractVector, y::AbstractVector,
    ; <keyword arguments>)
```

 
```{julia}
 using LSurvival
 using Random
 z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
 enter = zeros(length(t));
 m = fit(KMSurv, enter, t, d)
 mw = fit(KMSurv, enter, t, d, wts=wt)
```
 or, equivalently:

```{julia}
 kaplan_meier(enter, t, d, wts=wt)
```
"""

DOC_VARIANCE_KMSURV ="""
 Greenwood's formula for variance and confidence intervals of a Kaplan-Meier survival curve

```{julia}
 StatsBase.stderror(m::KMSurv)
```

```{julia}
 StatsBase.confint(m:KMSurv; level=0.95, method="normal")
```
 method:
   - "normal" normality-based confidence intervals
   - "lognlog" log(-log(S(t))) based confidence intervals


```{julia}
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

DOC_VARIANCE_AJSURV ="""
 Greenwood's formula for variance and confidence intervals of a Aalen-Johansen risk function

```{julia}
 res = z, x, outt, d, event, wts = LSurvival.dgm_comprisk(MersenneTwister(123123), 100)
 int = zeros(length(d)) # no late entry
 m = fit(AJSurv, int, outt, event)
 stderror(m)
 confint(m, level=0.95)
```
"""

DOC_FIT_AJSURV ="""
 Aalen-Johansen estimator for cumulative risk

```{julia}
 using LSurvival
 using Random
 z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
 enter = zeros(length(t));
    # event variable is coded 0[referent],1,2
 m = fit(AJSurv, enter, t, event)
 mw = fit(AJSurv, enter, t, event, wts=wt)
```
 or, equivalently:

```{julia}
 aalen_johansen(enter::AbstractVector, exit::AbstractVector, y::AbstractVector,
    ; <keyword arguments>)
```
"""


####### non-user functions

# TODO: UPDATE
DOC_LGH_BRESLOW ="""
Update the partial likelihood, gradient and Hessian values from a Cox model fit (used during fitting, not generally useful for users).

Uses Breslow's partial likelihood.

Updates over all observations

Signature

```{julia}
 lgh_breslow!(m::M, j, caseidx, risksetidx) where {M<:AbstractPH}
```
 
"""

DOC_LGH_EFRON ="""
Update the partial likelihood, gradient and Hessian values from a Cox model fit (used during fitting, not generally useful for users).

Uses Efron's partial likelihood.

Updates over all observations

Signature

```{julia}
 lgh_efron!(m::M, j, caseidx, risksetidx) where {M<:AbstractPH}
```
"""

DOC_LGH ="""
Update the partial likelihood, gradient and Hessian values from a Cox model fit (used during fitting, not generally useful for users).

Uses Breslow's or Efron's partial likelihood.

Updates over all a single observation. This is just a simple wrapper that calls `lgh_breslow!` or `lgh_efron!`

Signature

```{julia}
 lgh!(m::M, j, caseidx, risksetidx) where {M<:AbstractPH}
```
"""


DOC__UPDATE_PHPARMS ="""
Update the partial likelihood, gradient and Hessian values from a Cox model fit (used during fitting, not generally useful for users).

Uses Breslow's or Efron's partial likelihood.

Updates over all observations

```{julia}
 function _update_PHParms!(
  m::M,
  # big indexes
  ne::I,
  caseidxs::Vector{Vector{T}},
  risksetidxs::Vector{Vector{T}},
) where {M<:AbstractPH,I<:Int,T<:Int}
```
"""

DOC_FIT_PHSURV ="""
Survival curve estimation using multiple cox models (sometimes referred to as a multi-state model)


## Function Signatures

```{julia}
risk_from_coxphmodels(fitlist::Array{T}, args...; kwargs...) where {T<:PHModel}
```

```{julia}
fit(::Type{M}, fitlist::Vector{<:T}, ; fitargs...) where {M<:PHSurv,T<:PHModel}
```

## Optional keywords
- coef_vectors = nothing(default) or vector of coefficient vectors from the cox models [will default to the coefficients from fitlist models]
- pred_profile = nothing(default) or vector of specific predictor values of the same length as the coef_vectors[1]

```{julia}
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
 risk_from_coxphmodels([ft1,ft2], pred_profile=[mnx,mnz])
```
"""


####### data generation functions
DOC_DGM ="""
 Generating discrete survival data without competing risks

 Usage: dgm(rng, n, maxT;afun=int_0, yfun=yprob, lfun=lprob)
 dgm(n, maxT;afun=int_0, yfun=yprob, lfun=lprob)

 Where afun, yfun, and lfun are all functions that take arguments v,l,a and output time-specific values of a, y, and l respectively
 Example:

```{julia}

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

DOC_DGM_COMPRISK ="""
 Generating continuous survival data with competing risks

 Usage: dgm_comprisk(rng, n)
 dgm_comprisk(n)

         - rng = random number generator    
         - n = sample size

 Example:

```{julia}
 using LSurvival
     # 100 individuals with two competing events
 z,x,t,d,event,weights = LSurvival.dgm_comprisk(100)
     

```
"""

DOC_BOOTSTRAP_PHMODEL ="""
 Bootstrapping coefficients of a proportional hazards model

 Signatures

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




```{julia}
 using LSurvival, Random

 id, int, outt, data =
 LSurvival.dgm(MersenneTwister(1212), 500, 5; afun = LSurvival.int_0)

 d, X = data[:, 4], data[:, 1:3]
 weights = rand(length(d))

 # survival outcome:
 R = LSurvResp(int, outt, d, ID.(id))    # specification with ID only
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

```{julia}
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

DOC_BOOTSTRAP_KMSURV ="""
 Bootstrap methods for Kaplan-Meier survival curve estimator

  Signatures

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
   
   

```{julia}
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

DOC_BOOTSTRAP_AJSURV ="""
 Bootstrap methods for Aalen-Johansen cumulative risk estimator
  
  Signatures

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


```{julia}
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



######## DEPRECATED FUNCTIONS ###############
