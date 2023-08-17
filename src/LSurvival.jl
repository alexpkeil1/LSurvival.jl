module LSurvival
    ####### imports #######

    using Reexport
    using Distributions
    using LinearAlgebra
    using Random
    using Printf
    @reexport using StatsModels
    #
    using StatsBase
    import StatsBase: CoefTable, StatisticalModel, RegressionModel
    
    import StatsBase: coef, coeftable, coefnames, confint, 
                      #deviance, nulldeviance, dof, dof_residual,
                      loglikelihood, nullloglikelihood, nobs, stderror, vcov,
                      #residuals, predict, predict!,
                      fitted, fit, model_response, response, modelmatrix, PValue
    
    import Base.show
    
    ####### exports #######
    export 
        # types
        # Model types
        PHModel,
        KMSurv, 
        AJSurv, 
        PHParms,
        LSurvResp,
        LSurvCompResp,
        AbstractPH,
        AbstractNPSurv
        # Outcome types
        
    export 
        # functions    
        kaplan_meier,        # interface for estimating cumulative risk from non-parametric estimator
        aalen_johansen,      # interface for estimating cumulative risk from non-parametric competing risk estimator
        coxmodel,            # interface for Cox model
        cox_summary,         # convenience function to summarize Cox model results
        ci_from_coxmodels    # interface for estimating cumulative risk from hazard specific Cox models
        #subdistribution_hazard_cuminc
    export 
        coef, coeftable, coefnames, confint, 
        #deviance, nulldeviance, dof, dof_residual,
        loglikelihood, nullloglikelihood, #nobs, 
        stderror, vcov,
        #residuals, predict, predict!,
        fitted, fit, #model_response, response, modelmatrix, PValue
        coxph


    ####### Abstract types #######
    #abstract type LinPred end                         # linear predictor in statistical models
    #abstract type DensePred <: LinPred end            # linear predictor with dense X
    abstract type AbstractNPSurv end
    
    
"""
        AbstractLsurvResp

  Abstract type representing a model response vector
"""
  abstract type AbstractLSurvResp end                         
  abstract type AbstractLSurvParms end                         
  abstract type AbstractPH <: RegressionModel end   # model based on a linear predictor

    ####### function definitions #######
    
    include("shared_structs.jl")
    include("coxmodel.jl")
    include("npsurvival.jl")
    include("data_generators.jl")
    include("deprecated.jl")
    

end