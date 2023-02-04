using LSurvival
using Test

@testset "LSurvival.jl" begin
    id, int, outt, data = LSurvival.dgm(20, 5;afun=LSurvival.int_0);
  
    d,X = data[:,4], data[:,1:3];
    
    args = (int, outt, d, X);
    res = coxmodel(args..., method="efron");
    coxsum = cox_summary(res, alpha=0.05, verbose=true);  

    kaplan_meier(int, outt, d)
    #trivial case of non-competing events with late entry
    times_aj, surv, ajest, riskset, events = aalen_johansen(int, outt, d, dvalues=[1.0])
    times_sd, cumhaz, ci_sd = subdistribution_hazard_cuminc(int, outt, d, dvalues=[1.0])


    z,x,t,d, event,wt = LSurvival.dgm_comprisk(100);
  
    subdistribution_hazard_cuminc(zeros(length(t)), t, event, dvalues=[1.0, 2.0])
    subdistribution_hazard_cuminc(zeros(length(t)), t, event, dvalues=[1.0, 2.0], weights=wt)
    aalen_johansen(zeros(length(t)), t, event, dvalues=[1.0, 2.0])
    aalen_johansen(zeros(length(t)), t, event, dvalues=[1.0, 2.0], weights=wt)
    kaplan_meier(zeros(length(t)), t, d, weights=wt)
  
end
