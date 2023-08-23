#################################################################################################################### 
# Examples
####################################################################################################################
# you may need to install some additional packages
# import Pkg; Pkg.add("RCall")

using LSurvival, LinearAlgebra, RCall, BenchmarkTools, Random


###################################################################
# Fitting a basic Cox model, Kaplan-Meier curve using preferred functions
###################################################################
id, int, outt, data =
    LSurvival.dgm(MersenneTwister(123123), 100, 100; afun = LSurvival.int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))
wt ./= (sum(wt) / length(wt))

# equivalent methods for unweighted, default efron partial likelihood
fit(PHModel, X, int, outt, d)
coxph(X, int, outt, d)

# using Breslow partial likelihood, adding in weights, and setting higher tolerance
phfit = fit(PHModel, X, int, outt, d, wts = wt, ties = "breslow", rtol = 1e-13, atol = 1e-8, keepx=true, keepy=true)
basehaz!(phfit)

# extracting various bits from the model
coef(phfit)
stderror(phfit)
vcov(phfit)
loglikelihood(phfit)     # log partial likelihood
nullloglikelihood(phfit) # NOTE: this is the log partial likelihood at the initial values, not necessarily the null coefficient vector
fitted(phfit)

# setting initial values (note that null partial likelihood is likelihood at the starting values)
phfitinit = fit(PHModel, X, int, outt, d, wts = wt, start = [6.0, -2, 3.0])

# Kaplan-Meier curve
fit(KMSurv, int, outt, d, wts = wt)
kfit = kaplan_meier(int, outt, d, wts = wt)

fitted(kfit)
show(stdout, kfit, maxrows = 40)

###################################################################
# Competing risks: cause-specific Cox models, Aalen-Johansen estimator of cumulative incidence (risk)
###################################################################
res = z, x, outt, d, event, weights = LSurvival.dgm_comprisk(MersenneTwister(123123), 100)
int = zeros(length(d)) # no late entry
X = hcat(z, x)

# any event indicator (d) and event type (event)
hcat(d[1:15], event[1:15])

# cause-specific Cox models
ft1 = fit(PHModel, X, int, outt, d .* (event .== 1))
ft2 = fit(PHModel, X, int, outt, d .* (event .== 2))
ft2_equivalent = fit(PHModel, X, int, outt, (event .== 2))


# print likelihood verbosely
fit(PHModel, X, int, outt, d .* (event .== 2), verbose = true)
coxph(X, int, outt, d .* (event .== 2), verbose = true)


# Aalen-Johansen estimator (non-parametric) of marginal cause-specific risk in the sample
fit(AJSurv, int, outt, event)
risk2 = aalen_johansen(int, outt, event)


# two equivalent methods for generating cumulative incidence/risk from a Cox model 
# risk at the baseline values of each covariate (note high risk inferred by both covariates being protective of both event types)
fit(PHSurv, [ft1, ft2])
risk1_ref = risk_from_coxphmodels([ft1, ft2])

# risk at mean predictor values 
Xpred = mean(X, dims = 1)  # (a "profile" for the average individual)
risk1 = risk_from_coxphmodels(
    [ft1, ft2],
    coef_vectors = [coef(ft1), coef(ft2)],
    pred_profile = Xpred,
)

# risk at median predictor values 
Xpred_median = median(X, dims = 1)
risk1_median = risk_from_coxphmodels(
    [ft1, ft2],
    coef_vectors = [coef(ft1), coef(ft2)],
    pred_profile = Xpred_median,
)

# alternatively, you can give the referent meaningful values (e.g. by centering the predictors)
Xcen = X .- mean(X, dims = 1)
ft1c = fit(PHModel, Xcen, int, outt, d .* (event .== 1))
ft2c = fit(PHModel, Xcen, int, outt, d .* (event .== 2))
risk1c = risk_from_coxphmodels([ft1c, ft2c]) # compare with risk1 object

# You can also "trick" the risk_from_coxphmodels by using null cox models (coefficient is forced to be zero by using fit at initial values)
ft1d = fit(PHModel, X, int, outt, d .* (event .== 1), maxiter = 0, start = [0.0, 0.0])
ft2d = fit(PHModel, X, int, outt, d .* (event .== 2), maxiter = 0, start = [0.0, 0.0])
risk1d = risk_from_coxphmodels([ft1d, ft2d]) # compare with risk2 object



###################################################################
# Under the hood of some of the structs/models
###################################################################
# survival outcome:
R = LSurvResp(outt, d)         # specification if no late entry
R = LSurvResp(int, outt, d)    # specification with  late entry
R = LSurvResp(int, outt, d, weights)    # specification with  weights and late entry (no specification with weights and no late entry)

# PH model predictors
P = PHParms(X)

# not-yet-fit PH model object
M = PHModel(R, P, "breslow")
M = PHModel(R, P)  #default is "efron" method for ties
isfitted(M)  # confirm this is not yet "fitted"
LSurvival._fit!(M, start = [0.0, 0.0])
isfitted(M)

#################################################################################################################### 
# space savers, bootstrapping
####################################################################################################################
res = z, x, outt, d, event, weights = LSurvival.dgm_comprisk(MersenneTwister(123123), 800)
int = zeros(length(d)) # no late entry
X = hcat(z, x)

ft1big = fit(PHModel, X, int, outt, d .* (event .== 1), keepx = true, keepy = true)
ft1small = fit(PHModel, X, int, outt, d .* (event .== 1), keepx = false, keepy = false)


Base.summarysize(ft1big)   # 8447 bytes
Base.summarysize(ft1small) # 1591 bytes

# bootstrapping can only be done with fit objects where keepx=true, keepy=true
bootstrap(ft1big)      # give a single bootstrap iteration of a model (to be fit)
bootcoefs = bootstrap(MersenneTwister(123123), ft1big, 100) # draw 100 bootstrap samples of model coefficients

# StatsBase.std(bootcoefs, dims=1): 0.156726  0.109832
stderror(ft1big) # : 0.13009936549414425  0.11233304711033691



#################################################################################################################### 
# comparison with R survival package
####################################################################################################################
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


# now using Julia: check for likelihood evaluation at the point estimates from R
m = fit(
    PHModel,
    Matrix(cgd[:, [:height, :propylac]]),
    cgd.tstart,
    cgd.tstop,
    cgd.status,
    wts = cgd.weight,
    ties = "efron",
    rtol = 1e-12,
    atol = 1e-6,
    start = coxcoef,
    maxiter = 0,
);
m2 = fit(
    PHModel,
    Matrix(cgd[:, [:height, :propylac]]),
    cgd.tstart,
    cgd.tstop,
    cgd.status,
    wts = cgd.weight,
    ties = "breslow",
    rtol = 1e-12,
    atol = 1e-6,
    start = coxcoef2,
    maxiter = 0,
);

# efron likelihoods, weighted
[coxll[end], loglikelihood(m)]
# breslow likelihoods, weighted
[coxll2[end], loglikelihood(m2)]
# efron weighted gradient, weighted
hcat(ff, m.P._grad)
# breslow wt grad, weighted + unweighted look promising (float error?)
hcat(ff2, m2.P._grad)
# efron standard errors 
hcat(sqrt.(diag(coxvcov)), stderror(m))
# breslow covariance matrix
coxvcov2
vcov(m2)

# now fitting from start values of zero in Julia
m = fit(
    PHModel,
    Matrix(cgd[:, [:height, :propylac]]),
    cgd.tstart,
    cgd.tstop,
    cgd.status,
    wts = cgd.weight,
    ties = "efron",
    rtol = 1e-18,
    atol = 1e-9,
    start = zeros(2),
);
m2 = fit(
    PHModel,
    Matrix(cgd[:, [:height, :propylac]]),
    cgd.tstart,
    cgd.tstop,
    cgd.status,
    wts = cgd.weight,
    ties = "breslow",
    rtol = 1e-18,
    atol = 1e-9,
    start = zeros(2),
);


hcat(coxcoef, coef(m))
hcat(coxcoef2, coef(m2))
# efron likelihoods, weighted
[coxll[end], loglikelihood(m)]
# breslow likelihoods, weighted
[coxll2[end], loglikelihood(m2)]
# efron weighted gradient, weighted
hcat(ff, m.P._grad)
# breslow wt grad, weighted + unweighted look promising (float error?)
hcat(ff2, m2.P._grad)
# efron standard errors 
hcat(sqrt.(diag(coxvcov)), stderror(m))
# breslow covariance matrix
coxvcov2
vcov(m2)


###################################################################
# Time benchmarks using simulated data, calling R vs. using Julia
###################################################################


id, int, outt, data = LSurvival.dgm(MersenneTwister(), 1000, 100; afun = LSurvival.int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))
wt ./= (sum(wt) / length(wt))

beta, ll, g, h, basehaz =
    coxmodel(int, outt, d, X, weights = wt, method = "breslow", tol = 1e-9, inits = nothing);
beta2, ll2, g2, h2, basehaz2 =
    coxmodel(int, outt, d, X, weights = wt, method = "efron", tol = 1e-9, inits = nothing);
# fit(PHModel, X, int, outt, d, wts=wt, ties="breslow", start=[.9,.9,.9])
m = fit(PHModel, X, int, outt, d, wts = wt, ties = "breslow");
m2 = coxph(X, int, outt, d, wts = wt, ties = "efron");


# benchmark runtimes vs. calling R
function rfun(int, outt, d, X, wt)
    @rput int outt d X wt
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

# deprecated julia functions
function jfun(int, outt, d, X, wt)
    coxmodel(int, outt, d, X, weights = wt, method = "breslow", tol = 1e-9, inits = nothing)
end

function jfun2(int, outt, d, X, wt)
    fit(PHModel, X, int, outt, d, wts = wt, ties = "breslow", rtol = 1e-9)
end

@rput int outt d X wt;
tr2 = @btime rfun2(int, outt, d, X, wt);
tr = @btime rfun(int, outt, d, X, wt);
tj = @btime jfun(int, outt, d, X, wt);
tj2 = @btime jfun2(int, outt, d, X, wt);


# now with Efron's method
# benchmark runtimes vs. calling R
function rfun(int, outt, d, X, wt)
    @rput int outt d X wt
    R"""
       library(survival)
       df = data.frame(int=int, outt=outt, d=d, X=X)
       cfit = coxph(Surv(int,outt,d)~., weights=wt, data=df, ties="efron")
       coxcoefs_cr = coef(cfit)
    """
    @rget coxcoefs_cr
end
function rfun2(int, outt, d, X, wt)
    R"""
       library(survival)
       df = data.frame(int=int, outt=outt, d=d, X=X)
       cfit = coxph(Surv(int,outt,d)~., weights=wt, data=df, ties="efron")
       coxcoefs_cr = coef(cfit)
    """
end

# deprecated julia functions
function jfun(int, outt, d, X, wt)
    coxmodel(int, outt, d, X, weights = wt, method = "efron", tol = 1e-9, inits = nothing)
end

function jfun2(int, outt, d, X, wt)
    fit(PHModel, X, int, outt, d, wts = wt, ties = "efron", rtol = 1e-9)
end

@rput int outt d X wt;
tr2 = @btime rfun2(int, outt, d, X, wt);
tr = @btime rfun(int, outt, d, X, wt);
tj = @btime jfun(int, outt, d, X, wt);
tj2 = @btime jfun2(int, outt, d, X, wt);

###################################################################
# checking baseline hazard against R
###################################################################
using LSurvival, LinearAlgebra, RCall, BenchmarkTools, Random

id, int, outt, data = LSurvival.dgm(MersenneTwister(345), 100, 10; afun = LSurvival.int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))
wt ./= (sum(wt) / length(wt))
#wt ./ wt

m = fit(PHModel, X, int, outt, d, wts = wt, ties = "breslow", rtol = 1e-9, keepx=true, keepy=true)
m2 = fit(PHModel, X, int, outt, d, wts = wt, ties = "efron", rtol = 1e-9, keepx=true, keepy=true)




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
coxvcov2 = vcov(cfit2)
cfit
"""

@rget coxvcov;
@rget coxcoef;
@rget coxcoef2;
@rget coxll;
@rget bh;
@rget bh2;


hcat(
    diff(bh.hazard)[findall(diff(bh.hazard) .> floatmin())],
    m.bh[2:end, 1],
)
hcat(
    diff(bh2.hazard)[findall(diff(bh2.hazard) .> floatmin())],
    m2.bh[2:end, 1],
)

