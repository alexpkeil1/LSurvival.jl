####### module structs
DOC_PHSURV = """

"""

DOC_PHMODEL = """
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
"""
####### Primary methods

DOC_COXPH = """
coxph(X::AbstractMatrix, enter::AbstractVector, exit::AbstractVector, y::AbstractVector; <keyword arguments>)

Fit a generalized Cox proportional hazards model to data. Alias for `fit(PHModel, ...)`.

using LSurvival
using Random
z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
enter = zeros(length(t));
X = hcat(x,rand(length(x)));
m2 = fit(PHModel, X, enter, t, d, ties="breslow")
LSurvival.coxph(X, enter, t, d, ties="breslow")
coeftable(m)
"""


DOC_RISK_FROM_COXPHMODELS = """
Competing risk models:

Calculate survival curve and cumulative incidence (risk) function, get a set of Cox models (PHModel objects) that are exhaustive for the outcome types

fit(::Type{M},
fitlist::AbstractVector{<:T},
;
fitargs...) where {M<:PHSurv, T <: PHModel}

OR 

risk_from_coxphmodels(fitlist::Array{T}, args...; kwargs...) where T <: PHModel



fit for PHSurv objects

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
"""


####### generic methods

DOC_FIT_ABSTRACPH = """
fit(::Type{M},
X::AbstractMatrix,#{<:FP},
enter::AbstractVector{<:Real},
exit::AbstractVector{<:Real},
y::Union{AbstractVector{<:Real},BitVector}
;
ties = "breslow",
wts::AbstractVector{<:Real}      = similar(y, 0),
offset::AbstractVector{<:Real}   = similar(y, 0),
fitargs...) where {M<:AbstractPH}

using LSurvival
using Random
 z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 1000);
 enter = zeros(length(t));
 X = hcat(x,rand(length(x)));
 #R = LSurvResp(enter, t, Int64.(d), wt)
 #P = PHParms(X, "efron")
 #mod = PHModel(R,P, true)
 #_fit!(mod)
 m = fit(PHModel, X, enter, t, d, ties="efron")
 m2 = fit(PHModel, X, enter, t, d, ties="breslow")
 coeftable(m)
"""



####### non-user functions
DOC_LGH_BRESLOW = """
lgh_breslow!(_den, _LL, _grad, _hess, j, p, Xcases, Xriskset, _rcases, _rriskset, _wtcases, _wtriskset)

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
 lgh_breslow!(_den, _LL, _grad, _hess, j, p, Xcases, Xriskset, _rcases, _rriskset, _wtcases, _wtriskset)
"""


DOC_LGH_EFRON = """
lgh_efron!(_den, _LL, _grad, _hess, j, p, Xcases, X, _rcases, _r, _wtcases, _wt, caseidx, risksetidx)

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
"""


DOC_LGH = """
lgh!(lowermethod3,_den, _LL, _grad, _hess, j, p, X, _r, _wt, caseidx, risksetidx)

wrapper: calculate log partial likelihood, gradient, hessian contributions for a single risk set
          under a specified method for handling ties
(efron and breslow estimators only)
"""


DOC__STEPCOXi = """
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

wrapper: calculate log partial likelihood, gradient, hessian contributions across all risk sets
          under a specified method for handling ties (efron and breslow estimators only)

p = size(X,2)
_LL = zeros(1)
_grad = zeros(p)
_hess = zeros(p,p)
_den = zeros(1)
#
_B = rand(p)
eventtimes = sort(unique(_out[findall(d.==1)]))
"""