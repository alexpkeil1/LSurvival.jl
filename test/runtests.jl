using LSurvival
using Test
using Random

@testset "LSurvival.jl" begin
    
    id, int, outt, data = LSurvival.dgm(MersenneTwister(1212), 20, 5;afun=LSurvival.int_0);
  
    d,X = data[:,4], data[:,1:3];
    
    args = (int, outt, d, X);
    res = coxmodel(args..., method="efron");
    coxsum = cox_summary(res, alpha=0.05, verbose=true);  

    kaplan_meier(int, outt, d)
    #trivial case of non-competing events with late entry
    times_aj, surv, ajest, riskset, events = aalen_johansen(int, outt, d, dvalues=[1.0])
    #times_sd, cumhaz, ci_sd = subdistribution_hazard_cuminc(int, outt, d, dvalues=[1.0])


    z,x,t,d, event,wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 100);
  
    #subdistribution_hazard_cuminc(zeros(length(t)), t, event, dvalues=[1.0, 2.0])
    #subdistribution_hazard_cuminc(zeros(length(t)), t, event, dvalues=[1.0, 2.0], weights=wt)
    ajres= aalen_johansen(zeros(length(t)), t, event, dvalues=[1.0, 2.0])
    aalen_johansen(zeros(length(t)), t, event, dvalues=[1.0, 2.0], weights=wt)
    kms = kaplan_meier(zeros(length(t)), t, d, weights=wt)
  
    X = hcat(z,x)
    int = zeros(100)
    d1  = d .* Int.(event.== 1)
    d2  = d .* Int.(event.== 2)
    sum(d)/length(d)
        
    
    lnhr1, ll1, g1, h1, bh1 = coxmodel(int, t, d1, X, method="efron");
    lnhr2, ll2, g2, h2, bh2 = coxmodel(int, t, d2, X, method="efron");
    bhlist = [bh1, bh2]
    coeflist = [lnhr1, lnhr2]
    covarmat = sum(X, dims=1) ./ size(X,1)
    cires = ci_from_coxmodels(bhlist;eventtypes=[1,2], coeflist=coeflist, covarmat=covarmat)

    lnhr1, ll1, g1, h1, bh1 = coxmodel(int, t, d1, X, method="efron", maxiter=0);
    lnhr2, ll2, g2, h2, bh2 = coxmodel(int, t, d2, X, method="efron", maxiter=0);
    bhlist = [bh1, bh2]
    coeflist = [lnhr1, lnhr2]
    covarmat = sum(X, dims=1) ./ size(X,1)
    cires2 = ci_from_coxmodels(bhlist;eventtypes=[1,2], coeflist=coeflist, covarmat=covarmat)

    # 
    ajres[3]  # marginal CI
    cires[1] #  CI at specific covariate values
    cires2[1] #  marginal CI using Cox model trick


    # making a bunch of copies of data
    aa = [(int, t, d1, d2, event, X) for i in 1:30]

    bint, bt, bd1, bd2, bevent, bX = [reduce(vcat, map(x -> x[k], aa)) for k in 1:length(aa[1])]
    bajres= aalen_johansen(bint, bt, bevent, dvalues=[1.0, 2.0])

    
    ajres[3]  # marginal CI
    bajres[3]  # marginal CI, duplicated data (no impact on estimate)

    lnhr1b, ll1, g1, h1, bh1b = coxmodel(bint, bt, bd1, bX, method="efron", maxiter=0);
    lnhr2b, ll2, g2, h2, bh2b = coxmodel(bint, bt, bd2, bX, method="efron", maxiter=0);
    bhlist = [bh1b, bh2b]
    coeflist = [lnhr1b, lnhr2b]
    covarmat = sum(bX, dims=1) ./ size(bX,1)
    bcires2 = ci_from_coxmodels(bhlist;eventtypes=[1,2], coeflist=coeflist, covarmat=covarmat)


    cires2[1]
    bcires2[1]
   

end
