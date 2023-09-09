using Test
using LSurvival
using Random
#using DataFrames
#using BenchmarkTools # add during testing

@testset "LSurvival.jl" begin

    id, int, outt, data =
        LSurvival.dgm(MersenneTwister(112), 100, 10; afun = LSurvival.int_0)
    data[:, 1] = round.(data[:, 1], digits = 3)
    d, X = data[:, 4], data[:, 1:3]
    wt = rand(length(d))
    wt ./= (sum(wt) / length(wt))

    function jfun(int, outt, d, X, wt)
        coxmodel(
            int,
            outt,
            d,
            X,
            weights = wt,
            method = "breslow",
            tol = 1e-9,
            inits = nothing,
        )
    end

    function jfun2(int, outt, d, X, wt)
        fit(PHModel, X, int, outt, d, wts = wt, ties = "breslow", rtol = 1e-9)
    end
    println(jfun(int, outt, d, X, wt)[1:2])
    println(jfun2(int, outt, d, X, wt))

    tj = @time jfun(int, outt, d, X, wt)
    tj2 = @time jfun2(int, outt, d, X, wt)

    tj = @time jfun(int, outt, d, X, wt)
    tj2 = @time jfun2(int, outt, d, X, wt)

    tj = @time [jfun(int, outt, d, X, wt) for i = 1:10]
    tj2 = @time [jfun2(int, outt, d, X, wt) for i = 1:10]
    #
    res = fit(
        PHModel,
        X,
        int,
        outt,
        d,
        wts = wt,
        ties = "breslow",
        rtol = 1e-9,
        keepx = true,
        keepy = true,
    )

    id, int, outt, data =
        LSurvival.dgm(MersenneTwister(1212), 30, 5; afun = LSurvival.int_0)

    d, X = data[:, 4], data[:, 1:3]
    weights = rand(length(d))

    ft = coxph(X, int, outt, d)
    ft.RL = [ones(3,3)]
    println(ft)


    tab = (
        entertime = int,
        exittime = outt,
        death = data[:, 4] .== 1,
        x = data[:, 1],
        z1 = data[:, 2],
        z2 = data[:, 3],
    )

    f = @formula(Surv(entertime, exittime, death) ~ x + z1 + z2)
    ft = coxph(f, tab)
    println("formula fit")
    println(coefnames(ft))
    println(ft)

    println(coxph(@formula(Surv(entertime, exittime, death) ~ x + z1 + z2 + z1*x), tab, contrasts=Dict(:z1 => CategoricalTerm)))
    #without late entry
    println(coxph(@formula(Surv(exittime, death) ~ x + z1 + z2 + z1*x), tab, contrasts=Dict(:z1 => CategoricalTerm)))
    # without censoring
    println(coxph(@formula(Surv(exittime) ~ x + z1 + z2 + z1*x), tab, contrasts=Dict(:z1 => CategoricalTerm)))

    # survival outcome:
    LSurvResp([.5, .6], [1,0])
    LSurvResp([.5, .6], [1,0], origintime=0)
    LSurvCompResp([.5, .6], [1,0], origintime=0)

    R = LSurvResp(int, outt, d, ID.(id))    # specification with ID only
    print(R)
    R = LSurvResp(outt, d)         # specification if no late entry
    R = LSurvResp(int, outt, d)    # specification with  late entry
    @test all(R.wts .< 1.01)
    R = LSurvResp(int, outt, d, weights)    # specification with  weights and late entry (no specification with weights and no late entry)
    @test all(R.wts .== weights)

    # PH model predictors
    P = PHParms(X)
    @test all(size(P) .== size(X))

    # not-yet-fit PH model object
    M = PHModel(R, P, "breslow")
    @test M.ties == "breslow"
    M = PHModel(R, P)
    @test M.ties == "efron"

    #data = DataFrame(data, [:x,:z1,:z2,:y])
    # f = @formula(y~z1+z2+x)
    # contrasts = nothing
    # M = M
    # modelframe(f, data, contrasts, M)


    LSurvival._fit!(M, start = [0.0, 0.0, 0.0])

    R = LSurvResp(int, outt, d)
    R = LSurvResp(outt, d) # set all to zero
    #print(R)


    print(kaplan_meier(int, outt, d))
    #trivial case of non-competing events with late entry
    print(aalen_johansen(int, outt, d))

    (bootstrap(MersenneTwister(123), kaplan_meier(int, outt, d)))
    #trivial case of non-competing events with late entry
    (bootstrap(MersenneTwister(123), aalen_johansen(int, outt, d)))


    z, x, t, d, event, wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 100)
    enter = zeros(length(t))

    ajres = aalen_johansen(enter, t, event)
    bootstrap(MersenneTwister(123), ajres)
    ajres2 = aalen_johansen(enter, t, event, wts = wt)
    bootstrap(MersenneTwister(123), ajres2)
    kms = kaplan_meier(enter, t, d, wts = wt)

    #println(ajres)
    #println(stderror(ajres))
    (confint(ajres, level = 0.95))
    #println(kms)
    #println(stderror(kms))
    (confint(kms, level = 0.95))
    (confint(kms, level = 0.95, method = "lognlog"))

    X = hcat(z, x)
    d1 = d .* Int.(event .== 1)
    d2 = d .* Int.(event .== 2)

    ft2 = fit(PHModel, X, enter, t, (event .== 2), ties = "efron")
    ft1 = fit(
        PHModel,
        X,
        enter,
        t,
        (event .== 1),
        ties = "efron",
        verbose = true,
        keepx = true,
        keepy = true,
    )
    println(ft1)
    (bootstrap(MersenneTwister(123), ft1))
    (bootstrap(MersenneTwister(123), ft1, 2))
    coxph(X, enter, t, d2, ties = "efron")



    covarmat = sum(X, dims = 1) ./ size(X, 1)
    ciresb = risk_from_coxphmodels(
        [ft1, ft2];
        coef_vectors = [coef(ft1), coef(ft2)],
        pred_profile = covarmat,
    )


    ft1b = fit(
        PHModel,
        X,
        enter,
        t,
        d1,
        ties = "efron",
        maxiter = 0,
        id = [ID(i) for i in eachindex(t)],
    )
    ft2b = fit(PHModel, X, enter, t, d2, ties = "efron", maxiter = 0)
    covarmat = sum(X, dims = 1) ./ size(X, 1)
    cires2b = risk_from_coxphmodels(
        [ft1b, ft2b];
        coef_vectors = [coef(ft1b), coef(ft2b)],
        pred_profile = covarmat,
    )


    (cires2b)

##### residuals
dat3 = (
    time = [1,1,2,2,2,2,3,4,5],
    status = [1,0,1,1,1,0,0,1,0],
    x = [2,0,1,1,0,1,0,1,0],
    wt = [1,2,3,4,3,2,1,2,1]
)


ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="efron", maxiter=0)
res_est = residuals(ft, type="martingale")
res_true = [1-1/19,0-1/19,1-(1/19+10/48+20/114+10/84),1-(1/19+10/48+20/114+10/84),1-(1/19+10/48+20/114+10/84),0-(1/19+10/48+10/38+10/28),0-(1/19+10/48+10/38+10/28),1-(1/19+10/48+10/38+10/28+2/3),0-(1/19+10/48+10/38+10/28+2/3)]

@test all(isapprox.(res_est, res_true))

ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="breslow")
res_est = residuals(ft, type="martingale")
res_true = [0.85531,-0.02593,0.17636,0.17636,0.65131,-0.82364,-0.34869,-0.64894,-0.69808]

@test all(isapprox.(res_est, res_true, atol=0.00001))


dat3 = (
    time = [1,1,2,2,2,2,3,4,5],
    status = [1,0,1,1,1,0,0,1,0],
    x = [2,0,1,1,0,1,0,1,0],
    wt = [1,2,3,4,3,2,1,2,1]
)

ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="breslow", maxiter=0)
dM, dt, di = LSurvival.dexpected_NA(ft);
muX = LSurvival.muX_t(ft, di)
truemuX = [13/19,11/16,2/3]

@test all(isapprox.(muX, truemuX, atol=0.00001))


dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
)
ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow", maxiter=0)
resid = residuals(ft, type="martingale")
r = exp(ft.P._B[1])
truth = dat1.status .- [r/(3r+3), r/(3r+3), r/(3r+3) + 2r/(r+3), 1/(3r+3) + 2/(r+3), 1/(3r+3) + 2/(r+3), 1/(3r+3) + 2/(r+3) + 1]
@test all(isapprox.(resid, truth, atol=0.000001))

ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow")
resid = residuals(ft, type="martingale")
r = exp(ft.P._B[1])
truth = dat1.status .- [r/(3r+3), r/(3r+3), r/(3r+3) + 2r/(r+3), 1/(3r+3) + 2/(r+3), 1/(3r+3) + 2/(r+3), 1/(3r+3) + 2/(r+3) + 1]
@test all(isapprox.(resid, truth, atol=0.000001))


ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="efron", maxiter=0)
resid = residuals(ft, type="martingale")
r = exp(ft.P._B[1])
truth = dat1.status .- [r/(3r+3), r/(3r+3), r/(3r+3) + r/(r+3) + r/(r+5), 1/(3r+3) + 1/(r+3) + 1/(r+5), 1/(3r+3) + 1/(r+3) + 2/(r+5), 1/(3r+3) + 1/(r+3) + 2/(r+5) + 1]
@test all(isapprox.(resid, truth, atol=0.000001))

ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="efron")
resid = residuals(ft, type="martingale")
r = exp(ft.P._B[1])
truth = dat1.status .- [r/(3r+3), r/(3r+3), r/(3r+3) + r/(r+3) + r/(r+5), 1/(3r+3) + 1/(r+3) + 1/(r+5), 1/(3r+3) + 1/(r+3) + 2/(r+5), 1/(3r+3) + 1/(r+3) + 2/(r+5) + 1]
@test all(isapprox.(resid, truth, atol=0.000001))


######################################################################
dat2 = (
    enter = [1,2,5,2,1,7,3,4,8,8],
    exit = [2,3,6,7,8,9,9,9,14,17],
    status = [1,1,1,1,1,1,1,0,0,0],
    x = [1,0,0,1,0,1,1,1,0,0]
)

ft = coxph(@formula(Surv(enter, exit,status)~x),dat2, keepx=true, keepy=true, ties="breslow")
resid = residuals(ft, type="martingale")
truth = [0.521119,0.657411,0.789777,0.247388,-0.606293,0.369025,-0.068766,-1.068766,-0.420447,-0.420447]
@test all(isapprox.(resid, truth, atol=0.000001))


######################################################################
dat3 = (
    time = [1,1,2,2,2,2,3,4,5],
    status = [1,0,1,1,1,0,0,1,0],
    x = [2,0,1,1,0,1,0,1,0],
    wt = [1,2,3,4,3,2,1,2,1]
)

ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="breslow", maxiter=0)
resid = residuals(ft, type="martingale")
truth = [18/19,−1/19,49/152,49/152,49/152,−103/152,−103/152,−157/456,−613/456]
@test all(isapprox.(resid, truth, atol=0.000001))

ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="breslow")
resid = residuals(ft, type="martingale")
truth = [0.85531,-0.02593,0.17636,0.17636,0.65131,-0.82364,-0.34869,-0.64894,-0.69808]
@test all(isapprox.(resid, truth, atol=0.00001))


ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="efron", maxiter=0)
resid = residuals(ft, type="martingale")
truth = [1-1/19,0-1/19,1-(1/19+10/48+20/114+10/84),1-(1/19+10/48+20/114+10/84),1-(1/19+10/48+20/114+10/84),0-(1/19+10/48+10/38+10/28),0-(1/19+10/48+10/38+10/28),1-(1/19+10/48+10/38+10/28+2/3),0-(1/19+10/48+10/38+10/28+2/3)]
@test all(isapprox.(resid, truth, atol=0.000001))

ft = coxph(@formula(Surv(time,status)~x),dat3, wts=dat3.wt, keepx=true, keepy=true, ties="efron")
resid = residuals(ft, type="martingale")
truth = [0.8533453638920624,-0.025607157841240097,0.32265266060919384,0.32265266060919384,0.7169623432058416,-1.0777262895725614,-0.45034077190061034,-0.9049033864364076,-0.795986578172918]
@test all(isapprox.(resid, truth, atol=0.000001))


dat1 = (
    time = [1,1,6,6,8,9],
    status = [1,0,1,1,0,1],
    x = [1,1,1,0,0,0]
)
ft = coxph(@formula(Surv(time,status)~x),dat1, keepx=true, keepy=true, ties="breslow")
r = exp(ft.P._B[1])
# nXp matrix used for schoenfeld and score residuals
# (x(t)-mux(t))*dM(t) 
truthmat = permutedims(hcat(
  [(1-r/(r+1)) * (1-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), 0                         , 0],
  [(1-r/(r+1)) * (0-r/(3r+3)), (1-r/(r+3)) * (1-2r/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (1-2/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-2/(r+3)), 0],
  [(0-r/(r+1)) * (0-1/(3r+3)), (0-r/(r+3)) * (0-2/(r+3)), (0-0) * (1-1)]
))
truth = sum(truthmat, dims=2)[:]
S = residuals(ft, type="score")[:]
@test all(isapprox.(S, truth))
# assertions for testing breslow ties estimates under convergence for dat1
if length(ft.P._LL)>1
    @test isapprox(ft.P._B[1], 1.475285, atol=0.000001)
    @test isapprox(ft.P._LL[[1,end]], [-4.56434819, -3.82474951], atol=0.000001)
    @test isapprox(-ft.P._hess[1], 0.6341681, atol=0.000001)
end


end
