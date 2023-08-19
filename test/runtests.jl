using Test
using LSurvival
using Random

@testset "LSurvival.jl" begin

    id, int, outt, data =
        LSurvival.dgm(MersenneTwister(1212), 20, 5; afun = LSurvival.int_0)

    d, X = data[:, 4], data[:, 1:3]
    weights = rand(length(d))

    # survival outcome:
    R = LSurvResp(int, outt, d, ID.(id))    # specification with ID only
    R = LSurvResp(outt, d)         # specification if no late entry
    R = LSurvResp(int, outt, d)    # specification with  late entry
    @assert all(R.wts .< 1.01)
    R = LSurvResp(int, outt, d, weights)    # specification with  weights and late entry (no specification with weights and no late entry)
    @assert all(R.wts .== weights)

    # PH model predictors
    P = PHParms(X)

    # not-yet-fit PH model object
    M = PHModel(R, P, "breslow")
    @assert M.ties == "breslow"
    M = PHModel(R, P)
    @assert M.ties == "efron"

    R = LSurvResp(int, outt, d)
    R = LSurvResp(outt, d) # set all to zero
    print(R)


    args = (int, outt, d, X)
    res = coxmodel(args..., method = "efron")
    #coxsum = cox_summary(res, alpha=0.05, verbose=true);  

    kaplan_meier(int, outt, d)
    #trivial case of non-competing events with late entry
    aalen_johansen(int, outt, d)
    #times_sd, cumhaz, ci_sd = subdistribution_hazard_cuminc(int, outt, d, dvalues=[1.0])


    z, x, t, d, event, wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 100)
    enter = zeros(length(t))

    # resp = LSurvival.LSurvResp(enter, t, d, wt)
    #    X = hcat(z,x)
    #  coxmodel(resp.enter, resp.exit, Int.(resp.y), X, method="efron")

    #subdistribution_hazard_cuminc(zeros(length(t)), t, event, dvalues=[1.0, 2.0])
    #subdistribution_hazard_cuminc(zeros(length(t)), t, event, dvalues=[1.0, 2.0], weights=wt)
    ajres = aalen_johansen(enter, t, event)
    aalen_johansen(enter, t, event, wts = wt)
    kms = kaplan_meier(enter, t, d, wts = wt)

    println(ajres)
    println(kms)

    X = hcat(z, x)
    int = zeros(100)
    d1 = d .* Int.(event .== 1)
    d2 = d .* Int.(event .== 2)

    lnhr1, ll1, g1, h1, bh1 = coxmodel(int, t, d1, X, method = "efron")
    lnhr2, ll2, g2, h2, bh2 = coxmodel(int, t, d2, X, method = "efron")
    ft2 = fit(PHModel, X, int, t, (event .== 2), ties = "efron")
    ft1 = fit(PHModel, X, int, t, (event .== 1), ties = "efron", verbose = true)
    coxph(X, int, t, d2, ties = "efron")
    bhlist = [bh1, bh2]
    coeflist = [lnhr1, lnhr2]
    covarmat = sum(X, dims = 1) ./ size(X, 1)
    cires = ci_from_coxmodels(
        bhlist;
        eventtypes = [1, 2],
        coeflist = coeflist,
        covarmat = covarmat,
    )
    ciresb = risk_from_coxphmodels(
        [ft1, ft2];
        coef_vectors = [coef(ft1), coef(ft2)],
        pred_profile = covarmat,
    )


    lnhr1, ll1, g1, h1, bh1 = coxmodel(int, t, d1, X, method = "efron", maxiter = 0)
    lnhr2, ll2, g2, h2, bh2 = coxmodel(int, t, d2, X, method = "efron", maxiter = 0)
    ft1b = fit(PHModel, X, int, t, d1, ties = "efron", maxiter = 0)
    ft2b = fit(PHModel, X, int, t, d2, ties = "efron", maxiter = 0)
    bhlist = [bh1, bh2]
    coeflist = [lnhr1, lnhr2]
    covarmat = sum(X, dims = 1) ./ size(X, 1)
    cires2 = ci_from_coxmodels(
        bhlist;
        eventtypes = [1, 2],
        coeflist = coeflist,
        covarmat = covarmat,
    )
    cires2b = risk_from_coxphmodels(
        [ft1b, ft2b];
        coef_vectors = [coef(ft1b), coef(ft2b)],
        pred_profile = covarmat,
    )

    # 
    ajres  # marginal CI
    cires #  CI at specific covariate values
    cires2 #  marginal CI using Cox model trick
    cires2b #  marginal CI using Cox model trick


    println(ajres)
    println(cires2b)

    # making a bunch of copies of data
    aa = [(int, t, d1, d2, event, X) for i = 1:30]

    bint, bt, bd1, bd2, bevent, bX =
        [reduce(vcat, map(x -> x[k], aa)) for k = 1:length(aa[1])]
    bajres = aalen_johansen(bint, bt, bevent)



    ajres  # marginal CI
    bajres  # marginal CI, duplicated data (no impact on estimate)

    lnhr1b, ll1, g1, h1, bh1b = coxmodel(bint, bt, bd1, bX, method = "efron", maxiter = 0)
    lnhr2b, ll2, g2, h2, bh2b = coxmodel(bint, bt, bd2, bX, method = "efron", maxiter = 0)
    bhlist = [bh1b, bh2b]
    coeflist = [lnhr1b, lnhr2b]
    covarmat = sum(bX, dims = 1) ./ size(bX, 1)
    bcires2 = ci_from_coxmodels(
        bhlist;
        eventtypes = [1, 2],
        coeflist = coeflist,
        covarmat = covarmat,
    )


    print(cires2[1])
    print(bcires2[1])


end
