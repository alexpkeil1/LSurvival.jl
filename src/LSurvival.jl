module LSurvival
####### imports #######
using Documenter
using RecipesBase
using Reexport
using Printf
using Random, LinearAlgebra, Tables
#using Zygote
#using Distributions
#import Distributions: Chisq, Normal
import SpecialFunctions: gamma_inc, erfinv, erf, gamma, loggamma, digamma
#using SpecialFunctions
@reexport using StatsModels # ModelFrame, modelframe
#
#import DataFrames: DataFrame
using StatsBase
import StatsBase: CoefTable, StatisticalModel, RegressionModel
import Base: length, size, popat!, push!
#using Optim
import Optim: BFGS, optimize, Options, OnceDifferentiable, TwiceDifferentiable, only_fgh!, LineSearches, InitialHagerZhang

import StatsBase:
    aic,
    aicc,
    bic,
    coef,
    coeftable,
    coefnames,
    confint,
    deviance,
    nulldeviance, #, dof_residual,
    dof,
    fitted,
    fit,
    isfitted,
    loglikelihood,
    #lrtest,
    modelmatrix,
    model_response,
    nullloglikelihood,
    nobs,
    PValue,
    stderror,
    residuals,
    predict, 
    predict!,
    response,
    score,
    var,
    vcov,
    weights

import Base: convert, show

####### exports #######

# Structs
export AbstractPH,
    AbstractPSModel,
    AbstractNPSurv,
    AbstractLSurvivalID,
    AbstractLSurvivalParms,
    AbstractSurvDist,
    AbstractSurvTime,
    AJSurv,
    ID,
    KMSurv,
    LSurvivalResp,
    LSurvivalCompResp,
    PHModel,
    PSModel,
    PHParms,
    PSParms,
    PHSurv,
    # Strata
    Surv

# functions    
export kaplan_meier,        # interface for estimating cumulative risk from non-parametric estimator
    aalen_johansen,        # interface for estimating cumulative risk from non-parametric competing risk estimator
    coxph,                 # interface for Cox model
    survreg,                 # interface for parametric survival model
    risk_from_coxphmodels,  # interface for estimating cumulative risk from hazard specific Cox models
    # deprecated
    coxmodel,            # (deprecated) interface for Cox model
    cox_summary,         # (deprecated) convenience function to summarize Cox model results
    ci_from_coxmodels    # (deprecated) interface for estimating cumulative risk from hazard specific Cox models
#re-exports
export aic,
    aicc,
    bic,
    bootstrap,
    basehaz!,
    coef,
    coeftable,
    coefnames,
    confint,
    nulldeviance,
    deviance,
    dof,
    fit,
    fit!,
    fitted,
    lpdf,
    lsurv,
    model_response,
    modelmatrix, #PValue
    params,
    isfitted,
    jackknife,
    loglikelihood,
    logpartiallikelihood,
    lrtest, # re-exported
    modelmatrix,
    modelresponse,
    length,
    scale,
    shape,
    size,
    nullloglikelihood,
    nulllogpartiallikelihood,
    nobs,
    response,
    score,
    stderror,
    residuals,
    params,
    predict, 
    predict!,
    vcov,
    weights
# data
export survivaldata

####### Documentation #######
include("docstr.jl")


####### Abstract types #######

"""
$DOC_ABSTRACTLSURVRESP
"""
abstract type AbstractLSurvivalResp end

"""
$DOC_ABSTRACTLSURVPARMS
"""
abstract type AbstractLSurvivalParms end

"""
$DOC_ABSTRACTPH
"""
abstract type AbstractPH <: RegressionModel end   # model based on a linear predictor

"""
AbstractPS

Abstract type for parametric survival models
"""
abstract type AbstractPSModel <: RegressionModel end   # model based on a linear predictor

"""
$DOC_ABSTRACTNPSURV
"""
abstract type AbstractNPSurv end

"""
AbstractSurvDist

Abstract type for parametric survival distributions
"""
abstract type AbstractSurvDist end


####### function definitions #######

include("shared_structs.jl")
include("distributions.jl")
include("coxmodel.jl")
include("parsurvival.jl")
include("residuals.jl")
include("npsurvival.jl")
include("data_generators.jl")
include("bootstrap.jl")
include("jackknife.jl")
include("plot_recipes.jl")
include("example_data.jl")
include("deprecated.jl")



end