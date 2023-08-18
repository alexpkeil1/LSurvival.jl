#= #################################################################################################################### 
Examples
=# ####################################################################################################################

 
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
  