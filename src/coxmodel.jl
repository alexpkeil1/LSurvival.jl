#= #################################################################################################################### 
  Author: Alex Keil
  Program:  coxmodel.jl
  Language: Julia
  Creation Date: 02/2023
  Project: survival analysis
  Tasks: Newton raphson-algorithm to fit a Cox model via partial likelihood
  Description: Fit a cox model allowing for weights, late entry, right censoring, and time-varying covariates
    Both Efron's (default) and Breslow's method for ties are allowed.

  I wrote this because the Survival module in Julia does not support late entry or weights

=# ####################################################################################################################

#= #################################################################################################################### 
helper functions
=# ####################################################################################################################
calcp(z) = (1.0 - cdf(Distributions.Normal(), abs(z)))*2

function cox_summary(args; alpha=0.05, verbose=true)
  beta, ll, g, h, basehaz = args
  std_err = diag(-inv(h))
  z = beta./std_err
  zcrit = quantile.(Distributions.Normal(), [alpha/2.0, 1.0-alpha/2.0])
  lci = beta .+ zcrit[1]*std_err
  uci = beta .+ zcrit[2]*std_err
  pval = calcp.(z)
  op = hcat(beta, std_err, lci, uci, z, pval)
  verbose ? true : return(op)
  chi2 =  ll[end] - ll[1] 
  df = length(beta)
  lrtp = 1 - cdf(Distributions.Chisq(df), chi2)
  str = "Maximum partial likelihood estimates (alpha=$alpha):\n"
  str *= "-------------------------------------------------------\n"
  str *= "     ln(HR)   Std.Err LCI     UCI     Z       P(>|Z|)\n"
  str *= "-------------------------------------------------------"
  for (i,r) in enumerate(eachrow(op))
    str *= "\nb$i       "[1:6]
    str *= "$(r[1])       "[1:8]
    str *= " $(r[2])        "[1:8]
    str *= " $(r[3])        "[1:8]
    str *= " $(r[4])        "[1:8]
    str *= " $(r[5])        "[1:8]
    str *= " $(r[6])        "[1:8]
  end
  str *= "\n-------------------------------------------------------\n"
  str *= "Partial log-likelihood (null): $(ll[1])\n"
  str *= "Partial log-likelihood (fitted): $(ll[end])\n"
  str *= "LRT p-value (X^2=$(round(chi2, digits=2)), df=$df): $lrtp\n"
  str *= "Newton-Raphson iterations: $(length(ll)-1)\n"
  print(str)
  op
end

function containers(in, out, d, X, wt, inits)
  if size(out,1)==0
    throw("error in function call")
   end
  if isnothing(wt)
    wt = ones(size(in, 1))
  end
  @assert length(size(X))==2
  n,p = size(X)
  # indexes,counters
  eventtimes = sort(unique(out[findall(d .> 0)]))
  # containers
  _B = isnothing(inits) ? zeros(p) : copy(inits)
  _r = ones(Float64,n)
  _LL = zeros(1)
  _grad = zeros(p)
  _hess = zeros(p, p) #initialize
  (n,p, eventtimes,_B,_r,_LL,_grad,_hess, wt)
end

function tune(_LL, tol)
  totiter=0
  λ=1.0
  absdiff = tol*2.
  oldQ = floatmax()
  lastLL = -floatmax()
  converged = false
  totiter, λ, absdiff, oldQ, lastLL, converged, [_LL[1]]
end

function _coxrisk!(_r, X, B)
	map!(z -> exp(z),_r, X*B)
  nothing
end

function _coxrisk(X, B)
  _r = ones(size(X,1))
	_coxrisk!(_r, X, B)
end

function _zeroout!(_LL, _grad, _hess)
  _LL *= 0.0
  _grad *= 0.0
  _hess *= 0.0
  nothing
end



#= #################################################################################################################### 
partial likelihood/gradient/hessian functions for tied events
=# ####################################################################################################################

 """
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
 update_breslow!(_den, _LL, _grad, _hess, j, p, Xcases, Xriskset, _rcases, _rriskset, _wtcases, _wtriskset)
 """
function update_breslow!(_den, _LL, _grad, _hess, j, p, Xcases, Xriskset, _rcases, _rriskset, _wtcases, _wtriskset)
  den = sum(_rriskset.* _wtriskset)
  _LL .+= sum(_wtcases .* log.(_rcases)) .- log(den)*sum(_wtcases)
  #
  numg = Xriskset' * (_rriskset .* _wtriskset) 
  xbar = numg / den # risk-score-weighted average of X columns among risk set
  _grad .+= (Xcases .- xbar')' * (_wtcases)
  #
  numgg = (Xriskset' * Diagonal(_rriskset .* _wtriskset) * Xriskset) 
  xxbar = numgg/den  
  _hess .+=  - (xxbar -xbar*xbar') * sum(_wtcases)
  _den[j] = den
  nothing
  #updates (_ll, _grad, _hess)
end


function efron_weights(m)
  [(l-1)/m for l in 1:m]
end

"""
# for a given risk set
#compute log-likelihood, gradient vector and hessian matrix of cox model given individual level contriubtions
TODO: fix baseline hazard computations (using breslow's right now?)
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
update_efron!(_den, _LL, _grad, _hess, j, p, Xcases, Xriskset, _rcases, _rriskset,  _wtcases, _wtriskset)
"""
function update_efron!(_den, _LL, _grad, _hess, j, p, Xcases, Xriskset, _rcases, _rriskset,  _wtcases, _wtriskset)
  nties = size(Xcases,1)
  effwts = efron_weights(nties)
  den = sum(_wtriskset .* _rriskset)
  denc = sum(_wtcases .* _rcases)
  dens = [den - denc*ew for ew in effwts]
  # too low when weighted
  _LL .+= sum(_wtcases .* log.(_rcases)) .- sum(log.(dens)) * 1/nties * sum(_wtcases) # gives same answer as R with weights
  #
  numg = Xriskset' * (_wtriskset .* _rriskset) 
  numgs = [numg .- ew*Xcases'*(_wtcases .* _rcases) for ew in effwts]
  xbars = numgs ./ dens # risk-score-weighted average of X columns among risk set
  _grad .+= Xcases' *  _wtcases
  for i in 1:nties
    _grad .+= (-xbars[i]) * sum(_wtcases)/nties
  end
  numgg = (Xriskset' * Diagonal(_wtriskset .* _rriskset) * Xriskset)
  numggs = [numgg .- ew .* Xcases' * Diagonal(_wtcases .* _rcases) * Xcases for ew in effwts]
  xxbars = numggs ./ dens
  #
  for i in 1:nties
    _hess .-= (xxbars[i] - xbars[i]*xbars[i]') .* sum(_wtcases)/nties
  end
  _den[j] = den # using Breslow estimator
  nothing
  #updates: (_ll, _grad, _hess)
end

"""
wrapper: calculate log partial likelihood, gradient, hessian contributions for a single risk set
          under a specified method for handling ties
(efron and breslow estimators only)
"""
function update!(lowermethod3,_den, _LL, _grad, _hess, j, p, X, _r, _wt, caseidx, risksetidx)
  whichmeth = findfirst(lowermethod3 .== ["efr", "bre"])
  isnothing(whichmeth) ? throw("Method not recognized") : true
  if whichmeth == 1
     update_efron!(_den, _LL, _grad, _hess, j, p, X[caseidx,:],  X[risksetidx,:], _r[caseidx], _r[risksetidx], _wt[caseidx], _wt[risksetidx])
  elseif whichmeth == 2
    update_breslow!(_den, _LL, _grad, _hess, j, p, X[caseidx,:], X[risksetidx,:], _r[caseidx], _r[risksetidx], _wt[caseidx], _wt[risksetidx])
  end
end


# calculate log likelihood, gradient, hessian at set value of _B
"""
wrapper: calculate log partial likelihood, gradient, hessian contributions across all risk sets
          under a specified method for handling ties
(efron and breslow estimators only)

p = size(X,2)
_LL = zeros(1)
_grad = zeros(p)
_hess = zeros(p,p)
_den = zeros(1)
#
_B = rand(p)
eventtimes = sort(unique(_out[findall(d.==1)]))
"""
function _stepcox!(
         lowermethod3,
         # recycled parameters
         _LL::Vector, _grad::Vector, _hess::Matrix{Float64},
         # data
         _in::Vector, _out::Vector, d::Vector, X, _wt::Vector,
         # fixed parameters
         _B::Vector, 
         # indexs
         p, n, eventtimes,
         # containers
         _r::Vector
                  )
  _coxrisk!(_r, X, _B) # updates all elements of _r as exp(X*_B)
  _zeroout!(_LL, _grad, _hess)
  # loop over event times
  den,wtdriskset,wtdcases = zeros(length(eventtimes)), zeros(length(eventtimes)), zeros(length(eventtimes))
  @inbounds for (j,_outj) in enumerate(eventtimes)
    #j=13; _outj = eventtimes[j]
    risksetidx = findall((_in .< _outj) .&& (_out .>= _outj))
    caseidx = findall((_in .< _outj) .&& isapprox.(_out, _outj) .&& (d .> 0))
    update!(lowermethod3, den, _LL, _grad, _hess, j, p, X, _r, _wt, caseidx, risksetidx)
    wtdriskset[j] = sum(_wt[risksetidx])
    wtdcases[j] = sum(_wt[caseidx])
  end # (j,_outj)
  den, wtdriskset,wtdcases
end #function _stepcox!

#= #################################################################################################################### 
Newton raphson wrapper functions
=# ####################################################################################################################
function checkconverged(_grad, lastLL, _LL, oldQ, λ)
  Q = 0.5 * (_grad'*_grad) #modified step size if gradient increases
  likrat = abs(lastLL/_LL[1])
  absdiff = abs(lastLL-_LL[1])
  reldiff = max(likrat, inv(likrat)) -1.0
  converged = (reldiff < tol) || (absdiff < sqrt(tol))
  if Q > oldQ
    λ *= 0.8  # tempering
  else
    λ = min(2.0λ, 1.) # de-tempering
  end
  isnan(_LL[1]) ? throw("LL is NaN") : true
  Q, λ, converged
end

"""
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
function coxmodel(_in::Array{<:Real,1}, _out::Array{<:Real,1}, d::Array{<:Real,1}, X::Array{<:Real,2}; weights=nothing, method="efron", inits=nothing , tol=10e-9,maxiter=500)
   conts = containers(_in, _out, d, X, weights, inits)
   (n,p, eventtimes,_B,_r,_LL,_grad,_hess,weights) = conts
   lowermethod3 = lowercase(method[1:3])
   den, _wtriskset, _wtcase = _stepcox!(lowermethod3, 
      _LL, _grad, _hess,
      _in, _out, d, X, weights,
      _B, p, n, eventtimes,_r)
   # tuning params
   totiter, λ, absdiff, oldQ, lastLL, converged, _llhistory, = tune(_LL, tol)
  
  # repeat newton raphson steps until convergence or max iterations
  @inbounds while totiter<maxiter
    totiter +=1
    ######
    # update 
    #######
    Q, λ, converged = checkconverged!(_grad, lastLL, _LL, oldQ, λ)
    if converged
      break
    elseif abs(_LL[1]) != Inf
      _B .+= inv(-(_hess))*_grad.*λ # newton raphson step
      oldQ=Q
    else
       throw("log-likelihood is infinite")
    end
    lastLL = _LL[1]
    den, _ = _stepcox!(lowermethod3,
      _LL, _grad, _hess,
      _in, _out, d, X, weights,
      _B, p, n, eventtimes,_r)
    push!(_llhistory, _LL[1])
  end
  if totiter==maxiter
    @warn "Algorithm did not converge after $totiter iterations"
  end
  bh = [_wtcase ./ den _wtriskset _wtcase eventtimes]
  (_B, _llhistory, _grad, _hess, bh)
end
;
coxmodel(_out::Array{<:Real,1}, d::Array{<:Real,1}, X::Array{<:Real,2};kwargs...) = coxmodel(zeros(typeof(_out), length(_out)), _out, d, X;kwargs...)

#= #################################################################################################################### 
Examples
=# ####################################################################################################################


if false
  
  using LSurvival
  #=
  # comparison with R
  using RCall
  # commented out to avoid errors when trying to interpret macros without RCall dependency
  R"""
  library(survival)
  data(cgd, package="survival")
  cgd = cgd
  #cgd$weight = 1.0 # comment out for weighting
  # trick to always get naive.var to be loaded
  cgd$newid = 1:203
  cfit = coxph(Surv(tstart,tstop,status)~height+propylac, weight=cgd$weight, data=cgd, ties="efron", robust=TRUE, id=newid)
  cfit2 = coxph(Surv(tstart,tstop,status)~height+propylac, weight=cgd$weight, data=cgd, ties="breslow", robust=TRUE, id=newid)
  bh = basehaz(cfit, centered=FALSE)
  coxcoef = cfit$coefficients
  coxll = cfit$loglik
  coxvcov = cfit$naive.var
  ff = cfit$first
  coxcoef2 = cfit2$coefficients
  coxll2 = cfit2$loglik
  coxvcov2 = cfit2$naive.var
  ff2 = cfit2$first
  print(cfit)
  print(cfit2)
  """
  @rget cgd;
  @rget bh;
  @rget coxcoef;
  @rget coxll;
  @rget ff
  @rget coxvcov;
  @rget coxcoef2;
  @rget coxll2;
  @rget ff2;
  @rget coxvcov2;
    coxargs = (cgd.tstart, cgd.tstop, cgd.status, Matrix(cgd[:,[:height,:propylac]]));
    include("/Users/keilap/Projects/NCI/CPUM/ipw_policy/code/coxmodel.jl")
    bb, l, gg,hh,_ = coxmodel(coxargs...;weights=cgd.weight, method="efron", tol=1e-9, inits=coxcoef, maxiter=0);
    bb2, l2, gg2,hh2,_ = coxmodel(coxargs...,weights=cgd.weight, method="breslow", tol=1e-9, inits=coxcoef2, maxiter=0);
    # efron likelihoods, weighted + unweighted look promising (float error?)
    [l[end], coxll[end]] 
    # breslow likelihoods, weighted + unweighted look promising (float error?)
    [l2[end], coxll2[end]] 
    # efron weighted gradient, weighted + unweighted look promising (float error?)
     gg
     ff
     # breslow wt grad, weighted + unweighted look promising (float error?)
     gg2
     ff2 
     # efron hessian (unweighted only is ok)
    sqrt.(diag(-inv(hh)))
     sqrt.(diag(coxvcov))
     # breslow hessian (both ok - vcov)
    -inv(hh2)
     coxvcov2

  =#

  coxargs = (cgd.tstart, cgd.tstop, cgd.status, Matrix(cgd[:,[:height,:propylac]]));
  beta, ll, g, h, basehaz = coxmodel(coxargs...,weights=cgd.weight,method="breslow", tol=1e-18, inits=zeros(2));

  vcat(beta, coxcoef)
  lls = vcat(ll[end], coxll[end])
  argmax(lls)
  sqrt.(diag(-inv(h)))
  sqrt.(diag(coxvcov2))
  (coxvcov2)



  # new data comparing internal methods

  #####
  using Random, LSurvival
  id, int, outt, data = LSurvival.dgm(MersenneTwister(), 1000, 10;afun=LSurvival.int_0)
  data[:,1] = round.(  data[:,1] ,digits=3)
  d,X = data[:,4], data[:,1:3]
  wt = rand(length(d))
  wt ./= (sum(wt)/length(wt))
  
  #=
  using RCall, BenchmarkTools
  id, int, outt, data = LSurvival.dgm(MersenneTwister(), 1000, 10;afun=LSurvival.int_0)
  data[:,1] = round.(  data[:,1] ,digits=3)
  d,X = data[:,4], data[:,1:3]
  wt = rand(length(d))
  wt ./= (sum(wt)/length(wt))

  # benchmark vs. R
  function rfun(int, outt, d, X, wt)
      @rput int outt d X wt ;
      R"""
         library(survival)
         df = data.frame(int=int, outt=outt, d=d, X=X)
         cfit = coxph(Surv(int,outt,d)~., weights=wt, data=df, ties="breslow")
         coxcoefs_cr = coef(cfit)
      """
        @rget coxcoefs_cr 
  end

  function jfun(int, outt, d, X, wt)
    coxmodel(int, outt, d, X, weights=wt, method="breslow", tol=1e-9, inits=nothing);
  end

  @btime rfun(int, outt, d, X, wt)
  @btime jfun(int, outt, d, X, wt)


  @rput int outt d X wt
  R"""
  library(survival)
  df = data.frame(int=int, outt=outt, d=d, X=X)
  cfit = coxph(Surv(int,outt,d)~., weights=wt, data=df, ties="breslow")
  cfit2 = coxph(Surv(int,outt,d)~., weights=wt, data=df, ties="efron")
  bh = basehaz(cfit, centered=FALSE)
  bh2 = basehaz(cfit2, centered=FALSE)
  coxcoef = cfit$coefficients
  coxll = cfit$loglik
  coxvcov = vcov(cfit)
  cfit
  """
  @rget coxcoef;
  @rget coxll;
  @rget bh;
  @rget bh2;
  hcat(diff(bh.hazard), basehaz[2:end,1])
  hcat(diff(bh2.hazard), basehaz2[2:end,1])
  =#
  

  

  

end
