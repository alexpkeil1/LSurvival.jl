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
structs
=# ####################################################################################################################



  mutable struct PHParms{D <: AbstractMatrix, B <: AbstractVector, R <: AbstractVector, L <: AbstractVector, H <: AbstractMatrix, I <: Real} <: AbstractLSurvParms  
    X::D
    _B::B                        # coefficient vector
    _r::R                        # linear predictor/risk
    _LL::L                 # partial likelihood history
    _grad::B     # gradient vector
    _hess::H     # Hessian matrix
    n::I                     # number of observations
    p::I                     # number of parameters
#    function coxmodel(_in::Array{<:Real,1}, _out::Array{<:Real,1}, d::Array{<:Real,1}, X::Array{<:Real,2}; weights=nothing, method="efron", inits=nothing , tol=10e-9,maxiter=500)
  end
  
 function PHParms(X::D, _B::B, _r::R, _LL::L, _grad::B, _hess::H) where {D <: AbstractMatrix, B <: AbstractVector, R <: AbstractVector, L <: AbstractVector, H <: AbstractMatrix}
    n = length(_r)
    p = length(_B)   
   return PHParms(X, _B, _r, _LL, _grad, _hess, n, p)
  end

function PHParms(X::D) where {D <: AbstractMatrix}
  n,p = size(X)
  PHParms(X, fill(0.0, p), fill(0.0, n), zeros(Float64, 1), fill(0.0, p), fill(0.0, p, p))
end
 
  # PH model
  
 # abstract type AbstractPH <: RegressionModel end
      

mutable struct PHModel{G <: LSurvResp,L <: AbstractLSurvParms} <: AbstractPH  
     R::G        # Survival response
     P::L        # parameters
     ties::String
     fit::Bool
     bh::AbstractMatrix
end

function PHModel(R::G, P::L, ties::String, fit::Bool) where {G <: LSurvResp,L <: AbstractLSurvParms}
    return PHModel(R, P, ties, fit, zeros(Float64, length(R.eventtimes), 4))
end

function PHModel(R::G, P::L, ties::String) where {G <: LSurvResp,L <: AbstractLSurvParms}
    return PHModel(R, P, ties, false)
end

"""
   using LSurvival
   using Random
   #import LSurvival._stepcox!
    z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 100);
    enter = zeros(length(t));
    X = hcat(x,z);
    R = LSurvResp(enter, t, Int64.(d), wt)
    P = PHParms(X)
    mf = PHModel(R,P)
    _fit!(mf)
    
"""  
function PHModel(R::G, P::L) where {G <: LSurvResp,L <: AbstractLSurvParms}
    return PHModel(R, P, "efron")
end


function _fit!(m::PHModel;
               verbose::Bool=false,
               maxiter::Integer=500,
               minstepfac::Real=0.001,
               atol::Real=sqrt(1e-8),
               rtol::Real=1e-8,
               start=nothing,
               kwargs...
)
    m.P._B = start
    if haskey(kwargs, :ties)
        m.ties = kwargs[:ties]
    end
    method = m.ties
   #
   lowermethod3 = lowercase(method[1:3])
   # tuning params
   λ=1.0
   #
   totiter=0
   #
   oldQ = floatmax() 
   lastLL = -floatmax()
   risksetidxs, caseidxs = [], []
   @inbounds for _outj in m.R.eventtimes
     push!(risksetidxs, findall((m.R.enter .< _outj) .&& (m.R.exit .>= _outj)))
     push!(caseidxs, findall((m.R.y .> 0) .&& isapprox.(m.R.exit, _outj) .&& (m.R.enter .< _outj)))
   end
   den, _sumwtriskset, _sumwtcase = _stepcox!(lowermethod3, 
      m.P._LL, m.P._grad, m.P._hess,
      m.R.enter, m.R.exit, m.R.y, m.P.X, m.R.wts,
      m.P._B, m.P.p, m.P.n, m.R.eventtimes, m.P._r, 
      risksetidxs, caseidxs)
  _llhistory = [m.P._LL[1]] # if inits are zero, 2*(_llhistory[end] - _llhistory[1]) is the likelihood ratio test on all predictors
  # repeat newton raphson steps until convergence or max iterations
  while totiter<maxiter
    totiter +=1
    ######
    # update 
    #######
    Q = 0.5 * (m.P._grad'*m.P._grad) #modified step size if gradient increases
    likrat = (lastLL/m.P._LL[1])
    absdiff = abs(lastLL-m.P._LL[1])
    reldiff = max(likrat, inv(likrat)) - 1.0
    converged = (reldiff < atol) || (absdiff < rtol)
    if converged
      break
    end
    if Q > oldQ
      λ *= 0.5  # step-halving
    else
      λ = min(2.0λ, 1.) # de-halving
      #bestb = _B
      nothing
    end
    isnan(m.P._LL[1]) ? throw("Log-partial-likelihood is NaN") : true
    if abs(m.P._LL[1]) != Inf
      m.P._B .+= inv(-(m.P._hess))*m.P._grad.*λ # newton raphson step
      oldQ=Q
    else
       throw("log-partial-likelihood is infinite")
    end
    lastLL = m.P._LL[1]
    den, _, _ = _stepcox!(lowermethod3,
      m.P._LL, m.P._grad, m.P._hess,
      m.R.enter, m.R.exit, m.R.y, m.P.X, m.R.wts,
      m.P._B, m.P.p, m.P.n, m.R.eventtimes, m.P._r, 
      risksetidxs, caseidxs)
    push!(_llhistory, m.P._LL[1])
  end
  if totiter==maxiter
    @warn "Algorithm did not converge after $totiter iterations"
  end
  if lowermethod3 == "bre"
    m.bh = [_sumwtcase ./ den _sumwtriskset _sumwtcase m.R.eventtimes]
  elseif lowermethod3 == "efr"
    m.bh = [1.0 ./ den _sumwtriskset _sumwtcase m.R.eventtimes]
  end
  m.P._LL = _llhistory
  m
end

function StatsBase.fit!(m::AbstractPH;
                        verbose::Bool=false,
                        maxiter::Integer=500,
                        minstepfac::Real=0.001,
                        atol::Real=1e-6,
                        rtol::Real=1e-6,
                        start=nothing,
                        kwargs...)
    if haskey(kwargs, :maxIter)
        Base.depwarn("'maxIter' argument is deprecated, use 'maxiter' instead", :fit!)
        maxiter = kwargs[:maxIter]
    end
    if haskey(kwargs, :minStepFac)
        Base.depwarn("'minStepFac' argument is deprecated, use 'minstepfac' instead", :fit!)
        minstepfac = kwargs[:minStepFac]
    end
    if haskey(kwargs, :convTol)
        Base.depwarn("'convTol' argument is deprecated, use `atol` and `rtol` instead", :fit!)
        rtol = kwargs[:convTol]
    end
    if !issubset(keys(kwargs), (:maxIter, :minStepFac, :convTol))
        throw(ArgumentError("unsupported keyword argument"))
    end
    if haskey(kwargs, :tol)
        Base.depwarn("`tol` argument is deprecated, use `atol` and `rtol` instead", :fit!)
        rtol = kwargs[:tol]
    end

    start = isnothing(start) ? zeros(Float64, m.P.p) : start

    _fit!(m, verbose=verbose, maxiter=maxiter, minstepfac=minstepfac, atol=atol, rtol=rtol, start=start; kwargs...)
end


  """
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
  function fit(::Type{M},
      X::AbstractMatrix,#{<:FP},
      enter::AbstractVector{<:Real},
      exit::AbstractVector{<:Real},
      y::Union{AbstractVector{<:Real},BitVector}
      ;
      ties = "breslow",
      wts::AbstractVector{<:Real}      = similar(y, 0),
      offset::AbstractVector{<:Real}   = similar(y, 0),
      fitargs...) where {M<:AbstractPH}
      
      # Check that X and y have the same number of observations
      if size(X, 1) != size(y, 1)
          throw(DimensionMismatch("number of rows in X and y must match"))
      end
           
      R = LSurvResp(enter, exit, y, wts)
      P = PHParms(X)
 
      res = M(R,P, ties)
      
      return fit!(res; fitargs...)
  end

"""
    coxph(X::AbstractMatrix, enter::AbstractVector, exit::AbstractVector, y::AbstractVector,
        ; <keyword arguments>)

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
coxph(X, enter, exit, y, args...; kwargs...) = fit(PHModel, X, enter, exit, y, args...; kwargs...)


function StatsBase.coef(m::M) where {M <: AbstractPH}
  m.P._B
end

function StatsBase.vcov(m::M) where {M <: AbstractPH}
  -inv(m.P._hess)
end

function StatsBase.stderror(m::M) where {M <: AbstractPH}
  sqrt.(diag(vcov(m)))
end


function StatsBase.loglikelihood(m::M) where {M <: AbstractPH}
  m.P._LL[end]
end

function StatsBase.nullloglikelihood(m::M) where {M <: AbstractPH}
  m.P._LL[1]
end

function StatsBase.confint(m::M; level::Real=0.95) where {M <: AbstractPH}
  beta = coef(m)
  std_err = stderror(m)
  z = beta./std_err
  zcrit = quantile.(Distributions.Normal(), [(1-level)/2, 1-(1-level)/2])
  lci = beta .+ zcrit[1]*std_err
  uci = beta .+ zcrit[2]*std_err
  hcat(lci,uci)
end


function StatsBase.coeftable(m::M; level::Real=0.95) where {M <: AbstractPH}
  beta = coef(m)
  std_err = stderror(m)
  z = beta./std_err
  zcrit = quantile.(Distributions.Normal(), [(1-level)/2, 1-(1-level)/2])
  lci = beta .+ zcrit[1]*std_err
  uci = beta .+ zcrit[2]*std_err
  pval = calcp.(z)
  op = hcat(beta, std_err, lci, uci, z, pval)
  head = ["ln(HR)","StdErr","LCI","UCI","Z","P(>|Z|)"]
  rown = ["b$i" for i in 1:size(op)[1]]
  StatsBase.CoefTable(op, head, rown, 6,5 )
end


function Base.show(io::IO, m::M; level::Real=0.95) where {M <: AbstractPH}
  ll = loglikelihood(m)
  llnull = nullloglikelihood(m)
  chi2 = ll - llnull
  head = ["ln(HR)","StdErr","LCI","UCI","Z","P(>|Z|)"]
  #rown = ["b$i" for i in 1:size(op)[1]]
  coeftab = coeftable(m, level=level)
  df = length(coeftab.rownms)
  lrtp = 1 - cdf(Distributions.Chisq(df), chi2)
  iob = IOBuffer();
  println(iob, coeftab);
  str = """\nMaximum partial likelihood estimates (alpha=$(@sprintf("%.2g", 1-level))):\n"""
  str *= String(take!(iob))
  str *= "Partial log-likelihood (null): $(@sprintf("%8g", llnull))\n"
  str *= "Partial log-likelihood (fitted): $(@sprintf("%8g", ll))\n"
  str *= "LRT p-value (X^2=$(round(chi2, digits=2)), df=$df): $(@sprintf("%.5g", lrtp))\n"
  str *= "Newton-Raphson iterations: $(length(m.P._LL)-1)"
  println(io, str)
end


if false


  # in progress
    function fit(::Type{M},
               f::FormulaTerm,
               data;
               wts::AbstractVector{<:Real}      = similar(y, 0),
               offset::AbstractVector{<:Real}   = similar(y, 0),
               method::Symbol = :cholesky,
               dofit::Union{Bool, Nothing} = nothing,
               contrasts::AbstractDict{Symbol}=Dict{Symbol,Any}(),
               fitargs...) where {M<:AbstractPH}
  
      f, (y, X) = modelframe(f, data, contrasts, M)
  
      # Check that X and y have the same number of observations
      #if size(X, 1) != size(y, 1)
      #    throw(DimensionMismatch("number of rows in X and y must match"))
      #end
  
      #rr = GlmResp(y, d, l, off, wts)
      
      #res = M(rr, X, nothing, false)
      R = LSurvResp(enter, exit, y, wts)
      P = PHParms(X)
 
      res = M(R,P, ties)

      #return coxmodel(_in::Array{<:Real,1}, 
      #          _out::Array{<:Real,1}, 
      #          d::Array{<:Real,1}, 
      #          X::Array{<:Real,2}; weights=nothing, method="efron", inits=nothing , tol=10e-9,maxiter=500)
      return fit!(res; fitargs...)
  end
  
end

#= #################################################################################################################### 
helper functions
=# ####################################################################################################################

calcp(z) = (1.0 - cdf(Distributions.Normal(), abs(z)))*2


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
 lgh_breslow!(_den, _LL, _grad, _hess, j, p, Xcases, Xriskset, _rcases, _rriskset, _wtcases, _wtriskset)
 """
function lgh_breslow!(_den, _LL, _grad, _hess, j, p, Xcases, Xriskset, _rcases, _rriskset, _wtcases, _wtriskset)
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
lgh_efron!(_den, _LL, _grad, _hess, j, p, Xcases, X, _rcases, _r, _wtcases, _wt, caseidx, risksetidx)
"""
function lgh_efron!(_den, _LL, _grad, _hess, j, p, Xcases, X, _rcases, _r,  _wtcases, _wt, caseidx, risksetidx)
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
function lgh!(lowermethod3,_den, _LL, _grad, _hess, j, p, X, _r, _wt, caseidx, risksetidx)
  whichmeth = findfirst(lowermethod3 .== ["efr", "bre"])
  isnothing(whichmeth) ? throw("Method not recognized") : true
  if whichmeth == 1
    lgh_efron!(_den, _LL, _grad, _hess, j, p, X[caseidx,:], X, _r[caseidx], _r, _wt[caseidx], _wt, caseidx, risksetidx)
  elseif whichmeth == 2
    lgh_breslow!(_den, _LL, _grad, _hess, j, p, X[caseidx,:], X[risksetidx,:], _r[caseidx], _r[risksetidx], _wt[caseidx], _wt[risksetidx])
  end
end

# calculate log likelihood, gradient, hessian at set value of _B
"""
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
function _stepcox!(
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
    lgh!(lowermethod3, den, _LL, _grad, _hess, j, p, X, _r, _wt, caseidx, risksetidx)
    wtdriskset[j] = sum(_wt[risksetidx])
    wtdcases[j] = sum(_wt[caseidx])
  end # j
  den, wtdriskset,wtdcases
end #function _stepcox!



mutable struct PHSurv{G <: Array{T} where {T <: PHModel}}  <: AbstractNPSurv  
  fitlist::G        # Survival response
  eventtypes::AbstractVector
  times::AbstractVector
  surv::Vector{Float64}
  risk::Matrix{Float64}
  basehaz::Vector{Float64}
  event::Vector{Float64}
end

function PHSurv(fitlist::Array{T}, eventtypes) where {T <: PHModel}
  bhlist = [ft.bh for ft in fitlist]
  bhlist = [hcat(bh, fill(eventtypes[i], size(bh,1))) for (i,bh) in enumerate(bhlist)]
  bh = reduce(vcat, bhlist)
  sp = sortperm(bh[:,4])
  bh = bh[sp,:]
  ntimes::Int64 = size(bh,1)
  risk, surv = zeros(Float64, ntimes, length(eventtypes)), fill(1.0, ntimes)
  times = bh[:,4]
  event = bh[:,5]
  PHSurv(fitlist, eventtypes, times, surv, risk, bh[:,1], event)
end

function PHSurv(fitlist::Array{T}) where {T <: PHModel}
  eventtypes = collect(1:length(fitlist))
  PHSurv(fitlist, eventtypes)
end

function _fit!(m::M; coeflist=nothing, covarmat=nothing) where {M <: PHSurv}
  #function ci_from_coxmodels(bhlist;eventtypes=[1,2], coeflist=nothing, covarmat=nothing)
    hr = ones(Float64,length(m.eventtypes))
    ch::Float64 = 0.0
    lsurv::Float64 = 1.0
    if !isnothing(coeflist)
      @inbounds for (j,d) in enumerate(m.eventtypes)
        hr[j] = exp(dot(coefmat, coeflist[j]))
      end 
    end
    lci = zeros(length(m.eventtypes))
    @inbounds for i in 1:length(m.basehaz)
      @inbounds for (j,d) in enumerate(m.eventtypes)
        if m.event[i] == d
          m.basehaz[i] *= hr[j]                        # baseline hazard times hazard ratio
          m.risk[i,j] =  lci[j] + m.basehaz[i] * lsurv 
        else 
          m.risk[i,j] =  lci[j]
        end
      end
      ch += m.basehaz[i]
      m.surv[i] = exp(-ch)
      lsurv = m.surv[i]
      lci = m.risk[i,:]
    end
    m #ci, surv, bh[:,5], bh[:,4]
  end


  """
    fit(::Type{M},
        fitlist::AbstractVector{<:T},
        ;
        fitargs...) where {M<:PHSurv, T <: PHModel}

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
   res = fit(PHSurv, [ft1, ft2])
   
"""                     
    function fit(::Type{M},
        fitlist::AbstractVector{<:T},
        ;
        fitargs...) where {M<:PHSurv, T <: PHModel}
        
        res = M(fitlist)
        
        return fit!(res; fitargs...)
    end



    function Base.show(io::IO, m::M; maxrows=20) where {M <: PHSurv}
      types = m.eventtypes
      ev = ["# events (j=$jidx)" for (jidx, j) in enumerate(types)]
      rr = ["risk (j=$jidx)" for (jidx, j) in enumerate(types)]
      
      resmat = hcat(m.times, m.surv, m.event, m.basehaz, m.risk)
      head = ["time","survival","event type","cause-specific hazard", rr...]
      nr = size(resmat)[1]
      rown = ["$i" for i in 1:nr]
  
      op = CoefTable(resmat, head, rown)
      iob = IOBuffer();
      if nr < maxrows
        println(iob, op);
      else
        len = round(Int,maxrows/2)
        op1, op2 = deepcopy(op), deepcopy(op)
        op1.rownms = op1.rownms[1:len]
        op1.cols = [c[1:len] for c in op1.cols]
        op2.rownms = op2.rownms[(end-len):end]
        op2.cols = [c[(end-len):end] for c in op2.cols]
        println(iob, op1)
        println(iob, "...")
        println(iob, op2)
      end
      str = """\nCox-model based survival, risk\n"""
      str *= String(take!(iob))
      for (jidx, j) in enumerate(types)
        str *= "Number of events (j=$j): $(@sprintf("%8g", sum(m.events .== m.eventtypes[jidx])))\n"
      end
      str *= "Number of unique event times: $(@sprintf("%8g", length(m.time)))\n"
      println(io, str)
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
    
    m = fit(PHModel, Matrix(cgd[:,[:height,:propylac]]), cgd.tstart, cgd.tstop, cgd.status, wts=cgd.weight, ties="efron", rtol=1e-12, atol=1e-6)
    m2 = fit(PHModel, Matrix(cgd[:,[:height,:propylac]]), cgd.tstart, cgd.tstop, cgd.status, wts=cgd.weight, ties="breslow", rtol=1e-12, atol=1e-6)
    # efron likelihoods, weighted + unweighted look promising (float error?)
    [l[end], coxll[end], loglikelihood(m)] 
    # breslow likelihoods, weighted + unweighted look promising (float error?)
    [l2[end], coxll2[end], loglikelihood(m2)] 
    # efron weighted gradient, weighted + unweighted look promising (float error?)
     gg
     ff
     m.P._grad
     # breslow wt grad, weighted + unweighted look promising (float error?)
     gg2
     ff2 
     m2.P._grad
     # efron hessian (unweighted only is ok)
     sqrt.(diag(-inv(hh)))
     sqrt.(diag(coxvcov))
     stderror(m)
     # breslow hessian (both ok - vcov)
    -inv(hh2)
     coxvcov2
     vcov(m)

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
    # fit(PHModel, X, int, outt, d, wts=wt, ties="breslow", start=[.9,.9,.9])
    fit(PHModel, X, int, outt, d, wts=wt, ties="breslow")

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
    #coxmodel(int, outt, d, X, weights=wt, method="breslow", tol=1e-9, inits=nothing);
    fit(PHModel, X, int, outt, d, wts=wt, ties="breslow", rtol=1e-9)
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
    m = fit(PHModel, X, int, outt, d, wts=wt, ties="breslow", rtol=1e-9);
    m2 = fit(PHModel, X, int, outt, d, wts=wt, ties="efron", rtol=1e-9);




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
  hcat(diff(bh.hazard)[findall(diff(bh.hazard) .> floatmin())], basehaz[2:end,1], m.bh[2:end,1])
  hcat(diff(bh2.hazard)[findall(diff(bh2.hazard) .> floatmin())], basehaz2[2:end,1], m2.bh[2:end,1])
  hcat(diff(bh2.hazard)[findall(diff(bh2.hazard) .> floatmin())] ./  basehaz2[2:end,1], basehaz2[2:end,2:end])


  hcat(bh2.hazard[1:1], basehaz2[1:1,:], m2.bh[1:1,:])
  length(findall(outt .== 11 .&& d .== 1))
  =#
  

  

  

end
