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

function formatp(pvalue)
  if pvalue < eps(typeof(pvalue))
    return(@sprintf("<%4.4g", eps(typeof(pvalue))))
  end
  str1 = @sprintf("%4.5g", pvalue)
  str1
end

function cox_summary(args; alpha=0.05, verbose=true)
  beta, ll, g, h, basehaz = args
  std_err = sqrt.(diag(-inv(h)))
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
  str = """
Maximum partial likelihood estimates (alpha=$alpha):

---------------------------------------------------------
    ln(HR)  StdErr  LCI     UCI     Z       P(>|Z|)
---------------------------------------------------------"""
  for (i,r) in enumerate(eachrow(op))
    str *= "\nb$i       "[1:4]
    str *= """ $(@sprintf("%7g", r[1]))       """[1:8]
    str *= """ $(@sprintf("%7g", r[2]))        """[1:8]
    str *= """ $(@sprintf("%7g", r[3]))        """[1:8]
    str *= """ $(@sprintf("%7g", r[4]))        """[1:8]
    str *= """ $(@sprintf("%7g", r[5]))        """[1:8]
    str *= """ $(formatp(r[6]))        """
  end
  str *= "\n---------------------------------------------------------\n"
  str *= "Partial log-likelihood (null): $(@sprintf("%8g", ll[1]))\n"
  str *= "Partial log-likelihood (fitted): $(@sprintf("%8g", ll[end]))\n"
  str *= "LRT p-value (X^2=$(round(chi2, digits=2)), df=$df): $(@sprintf("%5g", lrtp))\n"
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
  #eventidx = findall(d .> 0)
  eventtimes = sort(unique(out[findall(d .> 0)]))
  #nevents = length(eventidx);
  # containers
  _B = isnothing(inits) ? zeros(p) : copy(inits)
  _r = zeros(Float64,n)
  #_basehaz = zeros(Float64, nevents) # baseline hazard estimate
  #_riskset = zeros(Int64, nevents) # baseline hazard estimate
  _LL = zeros(1)
  _grad = zeros(p)
  _hess = zeros(p, p) #initialize
  #(n,p,eventidx, eventtimes,nevents,_B,_r, _basehaz, _riskset,_LL,_grad,_hess)
  n,p, eventtimes,_B,_r,_LL,_grad,_hess
end


function _coxrisk!(_r, X, B)
	map!(z -> exp(z),_r, X*B)
  nothing
end

function _coxrisk(X, B)
  _r = ones(size(X,1))
	_coxrisk!(_r, X, B)
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
 LGH_breslow!(_den, _LL, _grad, _hess, j, p, Xcases, Xriskset, _rcases, _rriskset, _wtcases, _wtriskset)
 """
function LGH_breslow!(_den, _LL, _grad, _hess, j, p, Xcases, Xriskset, _rcases, _rriskset, _wtcases, _wtriskset)
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
  #(_ll, _grad, _hess)
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
LGH_efron!(_den, _LL, _grad, _hess, j, p, Xcases, X, _rcases, _r, _wtcases, _wt, caseidx, risksetidx)
"""
function LGH_efron!(_den, _LL, _grad, _hess, j, p, Xcases, X, _rcases, _r,  _wtcases, _wt, caseidx, risksetidx)
  # things to drop
  #risksetnotcaseidx = setdiff(risksetidx, caseidx) # move out
  #_rnotcase  = _r[risksetnotcaseidx] # move out
  
  # things to move out of here
  Xriskset = X[risksetidx,:]
  _rriskset = _r[risksetidx]
  _wtriskset = _wt[risksetidx]

  nties = length(caseidx)
  effwts = efron_weights(nties)
  den = sum(_wtriskset .* _rriskset)
  denc = sum(_wtcases .* _rcases)
  dens = [den - denc*ew for ew in effwts]
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
  #_den[j] = den # Breslow estimator
  sw = sum(_wtcases)
  aw = sw / nties
  _den[j] = 1.0 ./ sum( aw ./ dens) # using Efron estimator
  nothing
  #(_ll, _grad, _hess)
end

"""
wrapper: calculate log partial likelihood, gradient, hessian contributions for a single risk set
          under a specified method for handling ties
(efron and breslow estimators only)
"""
function LGH!(lowermethod3,_den, _LL, _grad, _hess, j, p, X, _r, _wt, caseidx, risksetidx)
  whichmeth = findfirst(lowermethod3 .== ["efr", "bre"])
  isnothing(whichmeth) ? throw("Method not recognized") : true
  if whichmeth == 1
    LGH_efron!(_den, _LL, _grad, _hess, j, p, X[caseidx,:], X, _r[caseidx], _r, _wt[caseidx], _wt, caseidx, risksetidx)
  elseif whichmeth == 2
    LGH_breslow!(_den, _LL, _grad, _hess, j, p, X[caseidx,:], X[risksetidx,:], _r[caseidx], _r[risksetidx], _wt[caseidx], _wt[risksetidx])
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
         _r::Vector,
         # big indexes
         risksetidxs, caseidxs
                  )
  _coxrisk!(_r, X, _B) # updates all elements of _r as exp(X*_B)
  # loop over event times
  ne = length(eventtimes)
  den,wtdriskset,wtdcases = zeros(ne), zeros(ne), zeros(ne)
  _LL .*= 0.0
  _grad .*= 0.0
  _hess .*= 0.0
  @inbounds for j in 1:ne
    #j=2; _outj = eventtimes[j]
    #risksetidx = findall((_in .< _outj) .&& (_out .>= _outj))
    #caseidx = findall((d .> 0) .&& isapprox.(_out, _outj) .&& (_in .< _outj))
    risksetidx =risksetidxs[j]
    caseidx = caseidxs[j]
    LGH!(lowermethod3, den, _LL, _grad, _hess, j, p, X, _r, _wt, caseidx, risksetidx)
    wtdriskset[j] = sum(_wt[risksetidx])
    wtdcases[j] = sum(_wt[caseidx])
  end # j
  den, wtdriskset,wtdcases
end #function _stepcox!

#= #################################################################################################################### 
Newton raphson wrapper function
=# ############################################################################################################d########

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
  #(_in::Array{Float64}, _out::Array{Float64}, d, X::Array{Float64,2}, _wt::Array{Float64})=args
  #### move #####
   if isnothing(weights)
    weights = ones(size(_in, 1))
   end
   if size(_out,1)==0
    throw("error in function call")
   end
   conts = containers(_in, _out, d, X, weights, inits)
   #(n,p,eventidx, eventtimes,nevents,_B,_r, _basehaz, _riskset,_LL,_grad,_hess) = conts
   (n,p, eventtimes,_B,_r,_LL,_grad,_hess) = conts
   #
   lowermethod3 = lowercase(method[1:3])
   # tuning params
   totiter=0
   ??=1.0
   absdiff = tol*2.
   oldQ = floatmax() 
   #bestb = _B
   lastLL = -floatmax()
   risksetidxs, caseidxs = [], []
   @inbounds for _outj in eventtimes
     push!(risksetidxs, findall((_in .< _outj) .&& (_out .>= _outj)))
     push!(caseidxs, findall((d .> 0) .&& isapprox.(_out, _outj) .&& (_in .< _outj)))
   end
   den, _sumwtriskset, _sumwtcase = _stepcox!(lowermethod3, 
      _LL, _grad, _hess,
      _in, _out, d, X, weights,
      _B, p, n, eventtimes,_r, 
      risksetidxs, caseidxs)
  _llhistory = [_LL[1]] # if inits are zero, 2*(_llhistory[end] - _llhistory[1]) is the likelihood ratio test on all predictors
  converged = false
  # repeat newton raphson steps until convergence or max iterations
  while totiter<maxiter
    totiter +=1
    ######
    # update 
    #######
    Q = 0.5 * (_grad'*_grad) #modified step size if gradient increases
    likrat = (lastLL/_LL[1])
    absdiff = abs(lastLL-_LL[1])
    reldiff = max(likrat, inv(likrat)) -1.0
    converged = (reldiff < tol) || (absdiff < sqrt(tol))
    if converged
      break
    end
    if Q > oldQ
      ?? *= 0.5  # step-halving
    else
      ?? = min(2.0??, 1.) # de-halving
      #bestb = _B
      nothing
    end
    isnan(_LL[1]) ? throw("Log-partial-likelihood is NaN") : true
    if abs(_LL[1]) != Inf
      _B .+= inv(-(_hess))*_grad.*?? # newton raphson step
      oldQ=Q
    else
       throw("log-partial-likelihood is infinite")
    end
    lastLL = _LL[1]
    den, _, _ = _stepcox!(lowermethod3,
      _LL, _grad, _hess,
      _in, _out, d, X, weights,
      _B, p, n, eventtimes,_r, 
      risksetidxs, caseidxs)
    push!(_llhistory, _LL[1])
  end
  if totiter==maxiter
    @warn "Algorithm did not converge after $totiter iterations"
  end
  if lowermethod3 == "bre"
    bh = [_sumwtcase ./ den _sumwtriskset _sumwtcase eventtimes]
  elseif lowermethod3 == "efr"
    bh = [1.0 ./ den _sumwtriskset _sumwtcase eventtimes]
  end
  (_B, _llhistory, _grad, _hess, bh)
end
;
coxmodel(_out::Array{<:Real,1}, d::Array{<:Real,1}, X::Array{<:Real,2};kwargs...) = coxmodel(zeros(typeof(_out), length(_out)), _out, d, X;kwargs...)


"""
  z,x,outt,d,event,weights = LSurvival.dgm_comprisk(100)
  X = hcat(z,x)
  int = zeros(100)
  d1  = d .* Int.(event.== 1)
  d2  = d .* Int.(event.== 2)
  sum(d)/length(d)
  
  
  lnhr1, ll1, g1, h1, bh1 = coxmodel(int, outt, d1, X, method="efron");
  lnhr2, ll2, g2, h2, bh2 = coxmodel(int, outt, d2, X, method="efron");
  bhlist = [bh1, bh2]
  coeflist = [lnhr1, lnhr2]
  covarmat = sum(X, dims=1) ./ size(X,1)
  ci, surv = ci_from_coxmodels(bhlist;eventtypes=[1,2], coeflist=coeflist, covarmat=covarmat)
  """
function ci_from_coxmodels(bhlist;eventtypes=[1,2], coeflist=nothing, covarmat=nothing)
  bhlist = [hcat(bh, fill(eventtypes[i], size(bh,1))) for (i,bh) in enumerate(bhlist)]
  bh = reduce(vcat, bhlist)
  sp = sortperm(bh[:,4])
  bh = bh[sp,:]
  ntimes::Int64 = size(bh,1)
  ci, surv, hr = zeros(Float64, ntimes, length(eventtypes)), fill(1.0, ntimes), zeros(Float64,length(eventtypes))
  ch::Float64 = 0.0
  lsurv::Float64 = 1.0
  if !isnothing(coeflist)
    @inbounds for (j,d) in enumerate(eventtypes)
      hr[j] = exp(dot(covarmat, coeflist[j]))
    end 
  end
  lci = zeros(length(eventtypes))
  @inbounds for i in 1:ntimes
    @inbounds for (j,d) in enumerate(eventtypes)
      if bh[i,5] == d
        bh[i,1] *= hr[j]
        ci[i,j] =  lci[j] + bh[i,1] * lsurv 
      else 
        ci[i,j] =  lci[j]
      end
    end
    ch += bh[i,1]
    surv[i] = exp(-ch)
    lsurv = surv[i]
    lci = ci[i,:]
  end
  ci, surv, bh[:,5], bh[:,4]
end


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
  beta, ll, g, h, basehaz = coxmodel(coxargs...,weights=cgd.weight,method="efron", tol=1e-18, inits=zeros(2));
  beta2, ll2, g2, h2, basehaz2 = coxmodel(coxargs...,weights=cgd.weight,method="breslow", tol=1e-18, inits=zeros(2));

  hcat(beta, coxcoef)
  hcat(beta2, coxcoef2)
  lls = vcat(ll[end], coxll[end])
  lls2 = vcat(ll2[end], coxll2[end])
  argmax(lls)
  argmax(lls2)
  hcat(sqrt.(diag(-inv(h))), sqrt.(diag(coxvcov)))
  hcat(sqrt.(diag(-inv(h2))), sqrt.(diag(coxvcov2)))
  hcat(-inv(h), coxvcov)
  hcat(-inv(h2), coxvcov2)



  # new data comparing internal methods

  #####
  using Random, LSurvival
  id, int, outt, data = LSurvival.dgm(MersenneTwister(), 1000, 10;afun=LSurvival.int_0)
  data[:,1] = round.(  data[:,1] ,digits=3)
  d,X = data[:,4], data[:,1:3]
  wt = rand(length(d))
  wt ./= (sum(wt)/length(wt))
  
  #=

  using RCall, BenchmarkTools, Random, LSurvival
  id, int, outt, data = LSurvival.dgm(MersenneTwister(), 1000, 100;afun=LSurvival.int_0)
  data[:,1] = round.(  data[:,1] ,digits=3)
  d,X = data[:,4], data[:,1:3]
  wt = rand(length(d))
  wt ./= (sum(wt)/length(wt))

    beta, ll, g, h, basehaz = coxmodel(int, outt, d, X, weights=wt, method="breslow", tol=1e-9, inits=nothing);
    beta2, ll2, g2, h2, basehaz2 = coxmodel(int, outt, d, X, weights=wt, method="efron", tol=1e-9, inits=nothing);


  # benchmark runtimes vs. calling R
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
  function rfun2(int, outt, d, X, wt)
      R"""
         library(survival)
         df = data.frame(int=int, outt=outt, d=d, X=X)
         cfit = coxph(Surv(int,outt,d)~., weights=wt, data=df, ties="breslow")
         coxcoefs_cr = coef(cfit)
      """
  end

  function jfun(int, outt, d, X, wt)
    coxmodel(int, outt, d, X, weights=wt, method="breslow", tol=1e-9, inits=nothing);
  end

  @rput int outt d X wt ;
  @btime rfun2(int, outt, d, X, wt);
  tr = @btime rfun(int, outt, d, X, wt);
  tj = @btime jfun(int, outt, d, X, wt);



  # checking baseline hazard against R
  using RCall, Random, LSurvival
  id, int, outt, data = LSurvival.dgm(MersenneTwister(), 100, 100;afun=LSurvival.int_0)
  data[:,1] = round.(  data[:,1] ,digits=3)
  d,X = data[:,4], data[:,1:3]
  wt = rand(length(d))
  wt ./= (sum(wt)/length(wt))
  #wt = wt ./ wt

    beta, ll, g, h, basehaz = coxmodel(int, outt, d, X, weights=wt, method="breslow", tol=1e-9, inits=nothing);
    beta2, ll2, g2, h2, basehaz2 = coxmodel(int, outt, d, X, weights=wt, method="efron", tol=1e-9, inits=nothing);




  @rput int outt d X wt
  R"""
  library(survival)
  df = data.frame(int=int, outt=outt, d=d, X=X)
  cfit = coxph(Surv(int,outt,d)~., weights=wt, data=df, ties="breslow")
  cfit2 = coxph(Surv(int,outt,d)~., weights=wt, data=df, ties="efron")
  bh = basehaz(cfit, centered=FALSE)
  bh2 = basehaz(cfit2, centered=FALSE)
  coxcoef = cfit$coefficients
  coxcoef2 = cfit2$coefficients
  coxll = cfit$loglik
  coxvcov = vcov(cfit)
  cfit
  """


  @rget coxcoef;
  @rget coxcoef2;
  @rget coxll;
  @rget bh;
  @rget bh2;
  hcat(diff(bh.hazard)[findall(diff(bh.hazard) .> floatmin())], basehaz[2:end,1])
  hcat(diff(bh2.hazard)[findall(diff(bh2.hazard) .> floatmin())], basehaz2[2:end,1])
  hcat(diff(bh2.hazard)[findall(diff(bh2.hazard) .> floatmin())] ./  basehaz2[2:end,1], basehaz2[2:end,2:end])


  hcat(bh2.hazard[1:1], basehaz2[1:1,:])
  length(findall(outt .== 11 .&& d .== 1))
  =#
  

  

  

end
