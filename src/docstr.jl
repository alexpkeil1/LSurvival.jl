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

Outcome type for survival outcome subject to left truncation and right censoring

```
struct LSurvResp{
E<:AbstractVector,
X<:AbstractVector,
Y<:AbstractVector,
W<:AbstractVector,
T<:Real,
I<:AbstractLSurvID,
} <: AbstractLSurvResp
enter::E
"`exit`: Time at observation end"
exit::X
"`y`: event occurrence in observation"
y::Y
"`wts`: observation weights"
wts::W
"`eventtimes`: unique event times"
eventtimes::E
"`origin`: origin on the time scale"
origin::T
"`id`: person level identifier (must be wrapped in ID() function)"
id::Vector{I}
end

```

```
function LSurvResp(
enter::E,
exit::X,
y::Y,
wts::W,
id::Vector{I},
) where {
E<:AbstractVector,
X<:AbstractVector,
Y<:AbstractVector,
W<:AbstractVector,
I<:AbstractLSurvID,
}
```

```
LSurvResp(
enter::E,
exit::X,
y::Y,
id::Vector{I},
) 

```

```
LSurvResp(
enter::E,
exit::X,
y::Y,
wts::W,
) 

```

```
LSurvResp(
enter::E,
exit::X,
y::Y,
)
```

```
LSurvResp(exit::X, y::Y) where {X<:AbstractVector,Y<:AbstractVector}
```
"""

DOC_LSURVCOMPRESP ="""

Outcome type for competing risk survival outcomes subject to left truncation and right censoring

```
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
"`id`: person level identifier (must be wrapped in ID() function)"
id::Vector{I}
"`eventtypes`: vector of unique event types"
eventtypes::V
"`eventmatrix`: matrix of indicators on the observation level"
eventmatrix::M
end

```

```
LSurvCompResp(
enter::E,
exit::X,
y::Y,
wts::W,
id::Vector{I}
)
```

```
LSurvCompResp(
enter::E,
exit::X,
y::Y,
id::Vector{I}
)
```

```
LSurvCompResp(
enter::E,
exit::X,
y::Y,
wts::W,
)
```

```
LSurvCompResp(
enter::E,
exit::X,
y::Y,
)
```
"""

DOC_PHMODEL ="""
PHModel: Mutable object type for proportional hazards regression

```
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

```{julia-repl}
using LSurvival
using Random
import LSurvival: _stepcox!, dgm_comprisk

z,x,t,d, event,wt = dgm_comprisk(MersenneTwister(1212), 100);
enter = zeros(length(t));
X = hcat(x,z);
R = LSurvResp(enter, t, Int64.(d), wt)
P = PHParms(X)
mf = PHModel(R,P)
 _fit!(mf)
```
"""

DOC_PHSURV ="""
Mutable type for proportional hazards models

PHSsurv: Object type for proportional hazards regression

Methods: fit, show

```
mutable struct PHSurv{G<:Array{T} where {T<:PHModel}} <: AbstractNPSurv
fitlist::G        # Survival response
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

Accepts any Number or String

```{julia-repl}
[ID(i) for i in 1:10]
```

Used for the id argument in 

  - Outcome types: LSurvResp, LSurvCompResp 
  - Model types: PHModel, KMRisk, AJRisk

"""

DOC_STRATA ="""
Type for identifying individuals in survival outcomes.

Accepts any Number or String

```{julia-repl}
[Strata(i) for i in 1:10]
```
Used for the strata argument in PHModel (not yet implemented)

"""
####### Primary methods

DOC_COXPH ="""
```
coxph(X::AbstractMatrix, enter::AbstractVector, exit::AbstractVector, y::AbstractVector; <keyword arguments>)
```

Fit a generalized Cox proportional hazards model to data. Alias for`fit(PHModel, ...)`.

```{julia-repl}
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

```
fit(::Type{M},
fitlist::AbstractVector{<:T},
;
fitargs...) where {M<:PHSurv, T <: PHModel}
```
OR 

```
risk_from_coxphmodels(fitlist::Array{T}, args...; kwargs...) where T <: PHModel

```


fit for PHSurv objects

```{julia-repl}
using LSurvival
using Random
z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
enter = zeros(length(t));
X = hcat(x,rand(length(x)));
#m2 = fit(PHModel, X, enter, t, d, ties="breslow")
ft1 = coxph(X, enter, t, d.*(event .== 1), ties="breslow");
ft2 = coxph(X, enter, t, d.*(event .== 2), ties="breslow");
fitlist = [ft1, ft2]
# these are equivalent
res = fit(PHSurv, [ft1, ft2])
res2 = risk_from_coxphmodels([ft1, ft2])
```
"""

DOC_E_YEARSOFLIFELOST ="""
Expected number of years of life lost due to cause k

```{julia-repl}
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
fit for AbstractPH objects

```
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

```{julia-repl}
using LSurvival
using Random
z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
enter = zeros(length(t));
X = hcat(x,rand(length(x)));
 #R = LSurvResp(enter, t, Int64.(d), wt)
 #P = PHParms(X,"efron")
 #mod = PHModel(R,P, true)
 #_fit!(mod)
m = fit(PHModel, X, enter, t, d, ties="efron")
m2 = fit(PHModel, X, enter, t, d, ties="breslow")
coeftable(m)
```
"""

DOC_FIT_KMSURV ="""
fit for KMSurv objects

```{julia-repl}
using LSurvival
using Random
z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
enter = zeros(length(t));
m = fit(KMSurv, enter, t, d)
mw = fit(KMSurv, enter, t, d, wts=wt)
```
or, equivalently:

```{julia-repl}
kaplan_meier(enter::AbstractVector, exit::AbstractVector, y::AbstractVector,
   ; <keyword arguments>)
```
"""

DOC_VARIANCE_KMSURV ="""
Greenwood's formula for variance and confidence intervals of a Kaplan-Meier survival curve

```
StatsBase.stderror(m::KMSurv)
```
```
StatsBase.confint(m:KMSurv; level=0.95, method="normal")
```
method:
  - "normal" normality-based confidence intervals
  - "lognlog" log(-log(S(t))) based confidence intervals


```{julia-repl}
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

```{julia-repl}
res = z, x, outt, d, event, wts = LSurvival.dgm_comprisk(MersenneTwister(123123), 100)
int = zeros(length(d)) # no late entry
m = fit(AJSurv, int, outt, event)
stderror(m)
confint(m, level=0.95)
```
"""

DOC_FIT_AJSURV ="""
Aalen-Johansen estimator for cumulative risk

```{julia-repl}
using LSurvival
using Random
z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
enter = zeros(length(t));
   # event variable is coded 0[referent],1,2
m = fit(AJSurv, enter, t, event)
mw = fit(AJSurv, enter, t, event, wts=wt)
```
or, equivalently:
```{julia-repl}
aalen_johansen(enter::AbstractVector, exit::AbstractVector, y::AbstractVector,
   ; <keyword arguments>)
```
"""


####### non-user functions
DOC_LGH_BRESLOW ="""
```
lgh_breslow!(_den, _LL, _grad, _hess, j, p, Xcases, Xriskset, _rcases, _rriskset, _wtcases, _wtriskset)
```
 # for a given risk set
 #compute log-likelihood, gradient vector and hessian matrix of cox model given individual level contriubtions

```{julia-repl}
Xcases=X[caseidx,:]
Xriskset=X[risksetidx,:]
 _rcases = _r[caseidx]
 _rriskset = _r[risksetidx]
 
 _wtcases=_wt[caseidx]
 _wtriskset=_wt[risksetidx]
p = size(X,2)
j = 1
 _LL = [0.0]
 _grad = zeros(p)
 _hess = zeros(p,p)
 _den = zeros(j)
lgh_breslow!(_den, _LL, _grad, _hess, j, p, Xcases, Xriskset, _rcases, _rriskset, _wtcases, _wtriskset)
```
"""


DOC_LGH_EFRON ="""
```
lgh_efron!(_den, _LL, _grad, _hess, j, p, Xcases, X, _rcases, _r, _wtcases, _wt, caseidx, risksetidx)
```

```{julia-repl}
# for a given risk set
#compute log-likelihood, gradient vector and hessian matrix of cox model given individual level contriubtions
Xcases=X[caseidx,:]
Xriskset=X[risksetidx,:]
_rcases = _r[caseidx]
_rriskset = _r[risksetidx]

_wtcases=_wt[caseidx]
_wtriskset=_wt[risksetidx]
p = size(X,2)
j = 1
_LL = [0.0]
_grad = zeros(p)
_hess = zeros(p,p)
_den = zeros(j)
lgh_efron!(_den, _LL, _grad, _hess, j, p, Xcases, X, _rcases, _r, _wtcases, _wt, caseidx, risksetidx)
```
"""


DOC_LGH ="""
```{julia-repl}
lgh!(lowermethod3,_den, _LL, _grad, _hess, j, p, X, _r, _wt, caseidx, risksetidx)
```
wrapper: calculate log partial likelihood, gradient, hessian contributions for a single risk set
under a specified method for handling ties
(efron and breslow estimators only)
"""


DOC__STEPCOXi ="""
calculate log likelihood, gradient, hessian at set value of coefficients

```{julia-repl}
_stepcox!(
lowermethod3,
    # recycled parameters
    _LL::Vector, _grad::Vector, _hess::Matrix{Float64},
    # data
    _in::Vector, _out::Vector, d::Union{Vector, BitVector}, X, _wt::Vector,
    # fixed parameters
    _B::Vector, 
    # indexs
p::T, n::U, eventtimes::Vector,
    # containers
    _r::Vector,
    # big indexes
risksetidxs, caseidxs
    ) where {T <: Int, U <: Int}
```
wrapper: calculate log partial likelihood, gradient, hessian contributions across all risk sets
under a specified method for handling ties (efron and breslow estimators only)
```{julia-repl}
p = size(X,2)
_LL = zeros(1)
_grad = zeros(p)
_hess = zeros(p,p)
_den = zeros(1)
#
_B = rand(p)
eventtimes = sort(unique(_out[findall(d.==1)]))
```
"""

DOC_FIT_PHSURV ="""
fit for AJSurv objects

```{julia-repl}
using LSurvival
using Random
z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
enter = zeros(length(t));
   # event variable is coded 0[referent],1,2
m = fit(AJSurv, enter, t, event)
mw = fit(AJSurv, enter, t, event, wts=wt)
```
or, equivalently:
```{julia-repl}
aalen_johansen(enter::AbstractVector, exit::AbstractVector, y::AbstractVector,
   ; <keyword arguments>)
```
"""


####### data generation functions
DOC_DGM ="""
Generating discrete survival data without competing risks

Usage: dgm(rng, n, maxT;afun=int_0, yfun=yprob, lfun=lprob)
dgm(n, maxT;afun=int_0, yfun=yprob, lfun=lprob)

Where afun, yfun, and lfun are all functions that take arguments v,l,a and output time-specific values of a, y, and l respectively
Example:
```{julia-repl}

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
```{julia-repl}
using LSurvival
    # 100 individuals with two competing events
z,x,t,d,event,weights = LSurvival.dgm_comprisk(100)
    

```
"""

DOC_BOOTSTRAP_PHMODEL ="""
Bootstrapping coefficients of a proportional hazards model

```
bootstrap(rng::MersenneTwister, m::PHModel)
```

```{julia-repl}
using LSurvival, Random

id, int, outt, data =
LSurvival.dgm(MersenneTwister(1212), 500, 5; afun = LSurvival.int_0)

d, X = data[:, 4], data[:, 1:3]
weights = rand(length(d))

# survival outcome:
R = LSurvResp(int, outt, d, ID.(id))    # specification with ID only
P = PHParms(X)

Mod = PHModel(R, P)
LSurvival._fit!(Mod, start=Mod.P._B)


# careful propogation of bootstrap sampling
idx, R2 = bootstrap(R)
P2 = bootstrap(idx, P)
Modb = PHModel(R2, P2)
LSurvival._fit!(Mod, start=Mod.P._B)

# convenience function for bootstrapping a model
Modc = bootstrap(Mod)
LSurvival._fit!(Modc, start=Modc.P._B)
Modc.P.X = nothing
Modc.R = nothing

```

bootstrap(rng::MersenneTwister, m::PHModel, iter::Int; kwargs...)

Bootstrap Cox model coefficients

```
LSurvival._fit!(mb, keepx=true, keepy=true, start=[0.0, 0.0])
```

```{julia-repl}
using LSurvival, Random
res = z, x, outt, d, event, wts = LSurvival.dgm_comprisk(MersenneTwister(123123), 100)
int = zeros(length(d)) # no late entry
X = hcat(z, x)

mainfit = fit(PHModel, X, int, outt, d .* (event .== 1), keepx=true, keepy=true)

mb = bootstrap(mainfit, 1000)
mainfit

```
"""

DOC_BOOTSTRAP_KMSURV ="""
```{julia-repl}
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
```{julia-repl}
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