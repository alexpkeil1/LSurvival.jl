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
function containers(in, out, d, X, wt, inits)
  @assert length(size(X))==2
  n,p = size(X)
  # indexes,counters
  eventidx = findall(d .> 0)
  eventtimes = sort(unique(out[findall(d .> 0)]))
  nevents = length(eventidx);
  # containers
  _B = isnothing(inits) ? zeros(p) : copy(inits)
  _r = zeros(Float64,n)
  _basehaz = zeros(Float64, nevents) # baseline hazard estimate
  _riskset = zeros(Int64, nevents) # baseline hazard estimate
  _LL = zeros(1)
  _grad = zeros(p)
  _hess = zeros(p, p) #initialize
  (n,p,eventidx, eventtimes,nevents,_B,_r, _basehaz, _riskset,_LL,_grad,_hess)
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
         _r::Vector
                  )
  _coxrisk!(_r, X, _B) # updates all elements of _r as exp(X*_B)
  # loop over event times
  den,wtdriskset,wtdcases = zeros(length(eventtimes)), zeros(length(eventtimes)), zeros(length(eventtimes))
  _LL .*= 0.0
  _grad .*= 0.0
  _hess .*= 0.0
  @inbounds for (j,_outj) in enumerate(eventtimes)
    #j=13; _outj = eventtimes[j]
    risksetidx = findall((_in .< _outj) .&& (_out .>= _outj))
    caseidx = findall((_in .< _outj) .&& isapprox.(_out, _outj) .&& (d .> 0))
    LGH!(lowermethod3, den, _LL, _grad, _hess, j, p, X, _r, _wt, caseidx, risksetidx)
    wtdriskset[j] = sum(_wt[risksetidx])
    wtdcases[j] = sum(_wt[caseidx])
  end # (j,_outj)
  den, wtdriskset,wtdcases
end #function _stepcox!

#= #################################################################################################################### 
Newton raphson wrapper function
=# ####################################################################################################################

"""
Estimate parameters of an extended Cox model

Using: Newton raphson algorithm with modified/adaptive step sizes

id, int, outt, data = dgm(MersenneTwister(), 1000, 10;regimefun=int_0)
d,X = data[:,4], data[:,1:3]

args = (int, outt, d, X, nothing)
beta, ll, g, h, basehaz = coxmodel(args, inits=nothing, tol=10e-4, maxiter=500)

Keyword inputs:
method="efron", 
inits=nothing , # initial parameter values, set to zero if this is nothing
tol=10e-9,      #  convergence tolerance based on log likelihod: likrat = abs(lastLL/_LL[1]), absdiff = abs(lastLL-_LL[1]), reldiff = max(likrat, inv(likrat)) -1.0
maxiter=500    # maximum number of iterations for Newton Raphson algorithm (set to zero to calculate likelihood, gradient, Hessian at the initial parameter values)

Outputs:
beta: coefficients 
ll: log partial likelihood history (all iterations)
g: gradient vector at MPLE
h: hessian matrix at MPLE
basehaz: Matrix: baseline hazard at referent level of all covariates, weighted risk set size, weighted # of cases, time

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
   (n,p,eventidx, eventtimes,nevents,_B,_r, _basehaz, _riskset,_LL,_grad,_hess) = conts
   #
   lowermethod3 = lowercase(method[1:3])
   # tuning params
   totiter=0
   λ=1.0
   #g = h = xn = ll = 0.
   absdiff = tol*2.
   oldQ = floatmax()
 
   bn1 = _B
   bestb = _B
   lastLL = -floatmax()
   den, _wtriskset, _wtcase = _stepcox!(lowermethod3, 
      _LL, _grad, _hess,
      _in, _out, d, X, weights,
      _B, p, n, eventtimes,_r)
  _llhistory = [_LL[1]] # if inits are zero, 2*(_llhistory[end] - _llhistory[1]) is the likelihood ratio test on all predictors
  converged = false
  # repeat newton raphson steps until convergence or max iterations
  while totiter<maxiter
    totiter +=1
    ######
    # update 
    #######
    Q = 0.5 * (_grad'*_grad) #modified step size if gradient increases
    likrat = abs(lastLL/_LL[1])
    absdiff = abs(lastLL-_LL[1])
    reldiff = max(likrat, inv(likrat)) -1.0
    converged = (reldiff < tol) || (absdiff < sqrt(tol))
    if converged
      break
    end
    if Q > oldQ
      λ *= 0.8  # tempering
    else
      λ = min(2.0λ, 1.) # de-tempering
      bestb = _B
    end
    isnan(_LL[1]) ? throw("LL is NaN") : true
    if abs(_LL[1]) != Inf
      _B .+= inv(-(_hess))*_grad.*λ # newton raphson step
      oldQ=Q
    else
       throw("log-likelihood is infinite")
    end
    lastLL = _LL[1]
    den, _, _ = _stepcox!(lowermethod3,
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
coxmodel(_out::Array{<:Real,1}, d::Array{<:Real,1}, X::Array{<:Real,2};kwargs...) = coxmodel(zeros(typeof(_out), length(_out)), _out, d, X;kwargs)

#= #################################################################################################################### 
Examples
=# ####################################################################################################################


if false
  include("/Users/keilap/Projects/NCI/CPUM/ipw_policy/code/coxmodel.jl")
  # comparison with R
  using RCall
  
  #=
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

  include("/Users/keilap/Projects/NCI/CPUM/ipw_policy/code/coxmodel.jl")

  coxargs = (cgd.tstart, cgd.tstop, cgd.status, Matrix(cgd[:,[:height,:propylac]]));
  beta, ll, g, h, basehaz = coxmodel(coxargs...,weights=cgd.weight,method="breslow", tol=1e-18, inits=zeros(2));

  vcat(beta, coxcoef)
  lls = vcat(ll[end], coxll[end])
  argmax(lls)
  sqrt.(diag(-inv(h)))
  sqrt.(diag(coxvcov2))
  (coxvcov2)



  # new data comparing internal methods
  include("/Users/keilap/Projects/NCI/CPUM/ipw_policy/code/coxmodel.jl")

  expit(mu) =  inv(1.0+exp(-mu))

  function int_nc(v,l,a)
    expit(-1.0 + 3*v + 2*l)
  end
  
  function int_0(v,l,a)
    0.1
  end
  
  function lprob(v,l,a)
    expit(-3 + 2*v + 0*l + 0*a)
  end

  function yprob(v,l,a)
    expit(-3 + 2*v + 0*l + 2*a)
  end
  
  function dgm(rng, n, maxT;regimefun=int_0)
    V = rand(rng, n)
    LAY = Array{Float64,2}(undef,n*maxT,4)
    keep = ones(Bool, n*maxT)
    id = sort(reduce(vcat, fill(collect(1:n), maxT)))
    time = (reduce(vcat, fill(collect(1:maxT), n)))
    for i in 1:n
      v=V[i]; l = 0; a=0;
      lkeep = true
      for t in 1:maxT
          currIDX = (i-1)*maxT + t
          l = lprob(v,l,a) > rand(rng) ? 1 : 0
          a = regimefun(v,l,a) > rand(rng) ? 1 : 0
          y = yprob(v,l,a) > rand(rng) ? 1 : 0
          LAY[currIDX,:] .= [v,l,a,y]
          keep[currIDX] = lkeep
          lkeep = (!lkeep || (y==1)) ? false : true
      end
    end 
    id[findall(keep)], time[findall(keep)] .- 1, time[findall(keep)],LAY[findall(keep),:]
  end

  #####
  using Random
  id, int, outt, data = dgm(MersenneTwister(), 1000, 10;regimefun=int_0)
  data[:,1] = round.(  data[:,1] ,digits=3)
  d,X = data[:,4], data[:,1:3]
  wt = rand(length(d))
  wt ./= (sum(wt)/length(wt))
  #wt = round.(wt,digits=3)
  sum(wt)


  include("/Users/keilap/Projects/NCI/CPUM/ipw_policy/code/coxmodel.jl")

  # checking baseline hazard
  coxargs = (int, outt, d, X);
  beta, ll, g, h, basehaz = coxmodel(coxargs..., weights=wt, method="breslow", tol=1e-9, inits=nothing);
  beta2, ll2, g2, h2, basehaz2 = coxmodel(coxargs..., weights=wt, method="efron", tol=1e-9, inits=nothing);
  se = sqrt.(diag(-inv(h)));
  hcat(beta, se, beta ./ se)



  #=
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