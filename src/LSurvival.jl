module LSurvival
####### imports #######

using Reexport
using Printf
using Random, Distributions, LinearAlgebra, Tables
@reexport using StatsModels # ModelFrame, modelframe
#
#import DataFrames: DataFrame
using StatsBase
import StatsBase: CoefTable, StatisticalModel, RegressionModel

import StatsBase:
    coef,
    coeftable,
    coefnames,
    confint,
    #deviance, nulldeviance, dof, dof_residual,
    fitted,
    fit,
    isfitted,
    loglikelihood,
    modelmatrix,
    model_response,
    nullloglikelihood,
    nobs,
    PValue,
    stderror,
    #residuals, predict, predict!,
    response,
    score,
    vcov,
    #
    weights

import Base: convert, show

####### exports #######
# types

# Model types
export ID,
    # Strata
    PHModel,
    KMSurv,
    AJSurv,
    PHSurv,
    PHParms,
    AbstractPH,
    AbstractNPSurv,
    AbstractLSurvID,
    AbstractLSurvParms
# Outcome types
export LSurvResp, LSurvCompResp
# functions    
export kaplan_meier,        # interface for estimating cumulative risk from non-parametric estimator
    aalen_johansen,        # interface for estimating cumulative risk from non-parametric competing risk estimator
    coxph,                 # interface for Cox model
    risk_from_coxphmodels,  # interface for estimating cumulative risk from hazard specific Cox models
    coxmodel,            # (deprecated) interface for Cox model
    cox_summary,         # (deprecated) convenience function to summarize Cox model results
    ci_from_coxmodels    # (deprecated) interface for estimating cumulative risk from hazard specific Cox models
#re-exports
export bootstrap,
    basehaz!,
    coef,
    coeftable,
    coefnames,
    confint,
    fit, #model_response, , modelmatrix, PValue
    fitted,
    isfitted,
    loglikelihood,
    modelmatrix,
    nullloglikelihood, #nobs, 
    response,
    risk_from_coxphmodels,
    score,
    stderror,
    #residuals, predict, predict!,
    vcov

####### Documentation #######
include("docstr.jl")


####### Abstract types #######

"""
$DOC_ABSTRACTLSURVRESP
"""
abstract type AbstractLSurvResp end

"""
$DOC_ABSTRACTLSURVPARMS
"""
abstract type AbstractLSurvParms end

"""
$DOC_ABSTRACTPH
"""
abstract type AbstractPH <: RegressionModel end   # model based on a linear predictor

"""
$DOC_ABSTRACTNPSURV
"""
abstract type AbstractNPSurv end

####### function definitions #######

include("shared_structs.jl")
include("coxmodel.jl")
include("npsurvival.jl")
include("data_generators.jl")
include("bootstrap.jl")
include("deprecated.jl")



end