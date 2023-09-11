using Test
using LSurvival
using Random
#using DataFrames
#using BenchmarkTools # add during testing

@testset "LSurvival.jl" begin
    println("Creating test datasets")
    dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])
    dat2 = (
        enter = [1, 2, 5, 2, 1, 7, 3, 4, 8, 8],
        exit = [2, 3, 6, 7, 8, 9, 9, 9, 14, 17],
        status = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        x = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
    )
    dat3 = (
        time = [1, 1, 2, 2, 2, 2, 3, 4, 5],
        status = [1, 0, 1, 1, 1, 0, 0, 1, 0],
        x = [2, 0, 1, 1, 0, 1, 0, 1, 0],
        wt = [1, 2, 3, 4, 3, 2, 1, 2, 1],
    )
    dat1clust = (
        id = [1, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        enter = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        exit = [1, 1, 1, 6, 1, 6, 1, 8, 1, 9],
        status = [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        x = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    )




    id, int, outt, data = LSurvival.dgm(MersenneTwister(112), 100, 10; afun = LSurvival.int_0)
    data[:, 1] = round.(data[:, 1], digits = 3)
    d, X = data[:, 4], data[:, 1:3]
    wt = rand(length(d))
    wt ./= (sum(wt) / length(wt))

    function jfun(int, outt, d, X, wt, i)
        i == 1 && println("Deprecated method")
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

    function jfun2(int, outt, d, X, wt, i)
        i == 1 && println("Stable method")
        fit(PHModel, X, int, outt, d, wts = wt, ties = "breslow", rtol = 1e-9)
    end

    println("Compilation times")
    tj = @time jfun(int, outt, d, X, wt, 1)
    tj2 = @time jfun2(int, outt, d, X, wt, 1)

    println("Compiled times")
    tj = @time jfun(int, outt, d, X, wt, 1)
    tj2 = @time jfun2(int, outt, d, X, wt, 1)

    println("10x Compiled times")
    tj = @time [jfun(int, outt, d, X, wt, i) for i = 1:10]
    tj2 = @time [jfun2(int, outt, d, X, wt, i) for i = 1:10]
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
    res2 = coxph(X, int, outt, d, wts = wt, ties = "breslow", rtol = 1e-9)


    @test isapprox(logpartiallikelihood(res), logpartiallikelihood(res2))
    @test isapprox(bic(res), bic(res2))


    id, int, outt, data = LSurvival.dgm(MersenneTwister(1212), 30, 5; afun = LSurvival.int_0)

    d, X = data[:, 4], data[:, 1:3]
    w = rand(length(d))
    # TESTs: do updated methods give the same answer as deprecated methods

    ft = coxph(X, int, outt, d, ties = "efron")
    ftb = coxph(X, int, outt, d, ties = "breslow")
    # TESTs: can RL be set?
    @test isnothing(ft.RL)
    ft.RL = [ones(3, 3)]
    @test ft.RL == [ones(3, 3)]

    # TESTs: can RL be re-set?
    try
        residuals(ft, type = "dfbeta")
    catch e
        @test typeof(e) == DimensionMismatch
    end

    oldbeta, _ = coxmodel(int, outt, d, X, method = "efron")
    oldbetab, _ = coxmodel(int, outt, d, X, method = "breslow")

    @test all(isapprox.(coef(ft), oldbeta))
    @test all(isapprox.(coef(ftb), oldbetab))


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
    # TEST: does print function work
    println(fit)
    # TEST: does formula get stored?
    @test !isnothing(ft.formula)

    #"formula fit"
    @test all(coefnames(ft) .== ["x", "z1", "z2"])
    f2 = @formula(Surv(entertime, exittime, death) ~ x + z1)
    f3 = @formula(Surv(entertime, exittime, death) ~ x)
    fnull = @formula(Surv(entertime, exittime, death) ~ x)
    ft2 = coxph(f2, tab)
    ft3 = coxph(f3, tab)
    ftnull = coxph(fnull, tab)
    println(lrtest(ft, ft3))





    (coxph(
        @formula(Surv(entertime, exittime, death) ~ x + z1 + z2 + z1 * x),
        tab,
        contrasts = Dict(:z1 => CategoricalTerm),
    ))
    #without late entry
    (coxph(
        @formula(Surv(exittime, death) ~ x + z1 + z2 + z1 * x),
        tab,
        contrasts = Dict(:z1 => CategoricalTerm),
    ))
    # without censoring
    (coxph(
        @formula(Surv(exittime) ~ x + z1 + z2 + z1 * x),
        tab,
        contrasts = Dict(:z1 => CategoricalTerm),
    ))

    # survival outcome:
    LSurvivalResp([0.5, 0.6], [1, 0])
    LSurvivalResp([0.5, 0.6], [1, 0], origintime = 0)
    LSurvivalCompResp([0.5, 0.6], [1, 0], origintime = 0)

    # TESTs: expected behavior of surv objects
    R = LSurvivalResp(int, outt, d, ID.(id))    # specification with ID only
    println(R)
    R = LSurvivalResp(outt, d)         # specification if no late entry
    R = LSurvivalResp(int, outt, d)    # specification with  late entry
    @test all(R.wts .< 1.01)
    R = LSurvivalResp(int, outt, d, w)    # specification with  weights and late entry (no specification with weights and no late entry)
    @test all(R.wts .== w)

    # PH model predictors
    P = PHParms(X)
    @test all(size(P) .== size(X))

    # TESTs: does ties argument default work?
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

    R = LSurvivalResp(int, outt, d)
    R = LSurvivalResp(outt, d) # set all to zero
    #println(R)

    # TESTs: do print functions work?
    println(kaplan_meier(int, outt, d))
    #trivial case of non-competing events with late entry
    println(aalen_johansen(int, outt, d))

    (bootstrap(MersenneTwister(123), kaplan_meier(int, outt, d)))
    #trivial case of non-competing events with late entry
    (bootstrap(MersenneTwister(123), aalen_johansen(int, outt, d)))


    z, x, t, d, event, wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 100)

    enter = zeros(length(t))


    # TESTs: do updated methods give the same answer as deprecated methods

    ajres = aalen_johansen(enter, t, event)
    bootstrap(MersenneTwister(123), ajres)
    ajres2 = aalen_johansen(enter, t, event, wts = wt)
    bootstrap(MersenneTwister(123), ajres2)
    kms = kaplan_meier(enter, t, d, wts = wt)


    _, oldsurv, _ = LSurvival.km(enter, t, d, weights = wt)
    @test isapprox(kms.surv, oldsurv[1:length(kms.surv)])

    a, oldsurvaj, oldFaj, c = LSurvival.aj(enter, t, event, weights = wt)
    @test all(isapprox.(ajres2.risk, 1.0 .- oldFaj[1:length(ajres2.times), :]))
    @test all(isapprox.(ajres2.surv, oldsurvaj[1:length(ajres2.times), :]))
    @test all(isapprox.(ajres2.surv, kms.surv))


    # TESTs: do the confidence intervals not include the estimate?
    (naj = confint(ajres, level = 0.95))
    @test all(ajres.risk[end, 1] .> naj[end, 1]) && all(ajres.risk[end, 1] .< naj[end, 2])
    @test all(ajres.risk[end, 2] .> naj[end, 3]) && all(ajres.risk[end, 2] .< naj[end, 4])

    # TEST: do the different confidence interval approaches yield different results?
    (n = confint(kms, level = 0.95))
    (ln = confint(kms, level = 0.95, method = "lognlog"))
    @test !(n == ln)

    # TEST: do the different confidence interval approaches not include the estimate?
    @test all(kms.surv .> n[:, 1]) && all(kms.surv .< n[:, 2])
    @test all(kms.surv .> ln[:, 1]) && all(kms.surv .< ln[:, 2])
    # TESTs: do the different confidence interval follow expected patterns?
    @test maximum(ln) < 1
    @test minimum(ln) > 0
    @test maximum(n) > maximum(ln)

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
    # TEST: does the length function work?
    @test length(ft1.R.id) == 100

    # TEST: does single bootstrap return an unfitted model?
    @test !(bootstrap(MersenneTwister(123), ft1)).fit

    # TEST: does multi bootstrap return intended length of results
    @test size(bootstrap(MersenneTwister(123), ft1, 3)) == (3, length(coef(ft1)))


    # TEST: do updated methods give the same answer as deprecated methods
    _, _, _, _, bh2 = coxmodel(enter, t, Int.(event .== 2), X)
    _, _, _, _, bh1 = coxmodel(enter, t, Int.(event .== 1), X)

    refrisk = risk_from_coxphmodels([ft1, ft2]).risk
    oldrisk, _ = ci_from_coxmodels([bh1, bh2])

    @test isapprox(refrisk, oldrisk, atol = 0.0001)


    # TEST: do predictions at covariate means change the risk?
    covarmat = sum(X, dims = 1) ./ size(X, 1)
    ciresb = risk_from_coxphmodels(
        [ft1, ft2];
        coef_vectors = [coef(ft1), coef(ft2)],
        pred_profile = covarmat,
    )

    @test all(refrisk[end, :] .> ciresb.risk[end, :])



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

    #########################################################################################################
    ##### residuals

    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat3,
        wts = dat3.wt,
        keepx = true,
        keepy = true,
        ties = "efron",
        maxiter = 0,
    )
    res_est = residuals(ft, type = "martingale")
    res_true = [
        1 - 1 / 19,
        0 - 1 / 19,
        1 - (1 / 19 + 10 / 48 + 20 / 114 + 10 / 84),
        1 - (1 / 19 + 10 / 48 + 20 / 114 + 10 / 84),
        1 - (1 / 19 + 10 / 48 + 20 / 114 + 10 / 84),
        0 - (1 / 19 + 10 / 48 + 10 / 38 + 10 / 28),
        0 - (1 / 19 + 10 / 48 + 10 / 38 + 10 / 28),
        1 - (1 / 19 + 10 / 48 + 10 / 38 + 10 / 28 + 2 / 3),
        0 - (1 / 19 + 10 / 48 + 10 / 38 + 10 / 28 + 2 / 3),
    ]

    @test all(isapprox.(res_est, res_true))

    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat3,
        wts = dat3.wt,
        keepx = true,
        keepy = true,
        ties = "breslow",
    )
    res_est = residuals(ft, type = "martingale")
    res_true = [
        0.85531,
        -0.02593,
        0.17636,
        0.17636,
        0.65131,
        -0.82364,
        -0.34869,
        -0.64894,
        -0.69808,
    ]

    @test all(isapprox.(res_est, res_true, atol = 0.00001))


    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat3,
        wts = dat3.wt,
        keepx = true,
        keepy = true,
        ties = "breslow",
        maxiter = 0,
    )
    dM, dt, di = LSurvival.dexpected_NA(ft)
    muX = LSurvival.muX_t(ft, di)
    truemuX = [13 / 19, 11 / 16, 2 / 3]

    @test all(isapprox.(muX, truemuX, atol = 0.00001))


    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat1,
        keepx = true,
        keepy = true,
        ties = "breslow",
        maxiter = 0,
    )
    resid = residuals(ft, type = "martingale")
    r = exp(ft.P._B[1])
    truth =
        dat1.status .- [
            r / (3r + 3),
            r / (3r + 3),
            r / (3r + 3) + 2r / (r + 3),
            1 / (3r + 3) + 2 / (r + 3),
            1 / (3r + 3) + 2 / (r + 3),
            1 / (3r + 3) + 2 / (r + 3) + 1,
        ]
    @test all(isapprox.(resid, truth, atol = 0.000001))

    show(ft.R.id)



    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat1,
        keepx = true,
        keepy = true,
        ties = "breslow",
    )
    resid = residuals(ft, type = "martingale")
    r = exp(ft.P._B[1])
    truth =
        dat1.status .- [
            r / (3r + 3),
            r / (3r + 3),
            r / (3r + 3) + 2r / (r + 3),
            1 / (3r + 3) + 2 / (r + 3),
            1 / (3r + 3) + 2 / (r + 3),
            1 / (3r + 3) + 2 / (r + 3) + 1,
        ]
    @test all(isapprox.(resid, truth, atol = 0.000001))


    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat1,
        keepx = true,
        keepy = true,
        ties = "efron",
        maxiter = 0,
    )
    resid = residuals(ft, type = "martingale")
    r = exp(ft.P._B[1])
    truth =
        dat1.status .- [
            r / (3r + 3),
            r / (3r + 3),
            r / (3r + 3) + r / (r + 3) + r / (r + 5),
            1 / (3r + 3) + 1 / (r + 3) + 1 / (r + 5),
            1 / (3r + 3) + 1 / (r + 3) + 2 / (r + 5),
            1 / (3r + 3) + 1 / (r + 3) + 2 / (r + 5) + 1,
        ]
    @test all(isapprox.(resid, truth, atol = 0.000001))

    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat1,
        keepx = true,
        keepy = true,
        ties = "efron",
    )
    resid = residuals(ft, type = "martingale")
    r = exp(ft.P._B[1])
    truth =
        dat1.status .- [
            r / (3r + 3),
            r / (3r + 3),
            r / (3r + 3) + r / (r + 3) + r / (r + 5),
            1 / (3r + 3) + 1 / (r + 3) + 1 / (r + 5),
            1 / (3r + 3) + 1 / (r + 3) + 2 / (r + 5),
            1 / (3r + 3) + 1 / (r + 3) + 2 / (r + 5) + 1,
        ]
    @test all(isapprox.(resid, truth, atol = 0.000001))


    ######################################################################
    # TESTING DAT 2
    ft = coxph(
        @formula(Surv(enter, exit, status) ~ x),
        dat2,
        keepx = true,
        keepy = true,
        ties = "breslow",
    )
    resid = residuals(ft, type = "martingale")
    truth = [
        0.521119,
        0.657411,
        0.789777,
        0.247388,
        -0.606293,
        0.369025,
        -0.068766,
        -1.068766,
        -0.420447,
        -0.420447,
    ]
    @test all(isapprox.(resid, truth, atol = 0.000001))


    ######################################################################
    # TESTING DAT 3

    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat3,
        wts = dat3.wt,
        keepx = true,
        keepy = true,
        ties = "breslow",
        maxiter = 0,
    )
    resid = residuals(ft, type = "martingale")
    truth = [
        18 / 19,
        1 / -19,
        49 / 152,
        49 / 152,
        49 / 152,
        103 / -152,
        103 / -152,
        157 / -456,
        613 / -456,
    ]
    @test all(isapprox.(resid, truth, atol = 0.000001))

    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat3,
        wts = dat3.wt,
        keepx = true,
        keepy = true,
        ties = "breslow",
    )
    resid = residuals(ft, type = "martingale")
    truth = [
        0.85531,
        -0.02593,
        0.17636,
        0.17636,
        0.65131,
        -0.82364,
        -0.34869,
        -0.64894,
        -0.69808,
    ]
    @test all(isapprox.(resid, truth, atol = 0.00001))


    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat3,
        wts = dat3.wt,
        keepx = true,
        keepy = true,
        ties = "efron",
        maxiter = 0,
    )
    resid = residuals(ft, type = "martingale")
    truth = [
        1 - 1 / 19,
        0 - 1 / 19,
        1 - (1 / 19 + 10 / 48 + 20 / 114 + 10 / 84),
        1 - (1 / 19 + 10 / 48 + 20 / 114 + 10 / 84),
        1 - (1 / 19 + 10 / 48 + 20 / 114 + 10 / 84),
        0 - (1 / 19 + 10 / 48 + 10 / 38 + 10 / 28),
        0 - (1 / 19 + 10 / 48 + 10 / 38 + 10 / 28),
        1 - (1 / 19 + 10 / 48 + 10 / 38 + 10 / 28 + 2 / 3),
        0 - (1 / 19 + 10 / 48 + 10 / 38 + 10 / 28 + 2 / 3),
    ]
    @test all(isapprox.(resid, truth, atol = 0.000001))

    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat3,
        wts = dat3.wt,
        keepx = true,
        keepy = true,
        ties = "efron",
    )
    resid = residuals(ft, type = "martingale")
    truth = [
        0.8533453638920624,
        -0.025607157841240097,
        0.32265266060919384,
        0.32265266060919384,
        0.7169623432058416,
        -1.0777262895725614,
        -0.45034077190061034,
        -0.9049033864364076,
        -0.795986578172918,
    ]
    @test all(isapprox.(resid, truth, atol = 0.000001))


    dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])
    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat1,
        keepx = true,
        keepy = true,
        ties = "breslow",
    )
    r = exp(ft.P._B[1])
    # nXp matrix used for schoenfeld and score residuals
    # (x(t)-mux(t))*dM(t) 
    truthmat = permutedims(
        hcat(
            [(1 - r / (r + 1)) * (1 - r / (3r + 3)), 0, 0],
            [(1 - r / (r + 1)) * (0 - r / (3r + 3)), 0, 0],
            [
                (1 - r / (r + 1)) * (0 - r / (3r + 3)),
                (1 - r / (r + 3)) * (1 - 2r / (r + 3)),
                0,
            ],
            [
                (0 - r / (r + 1)) * (0 - 1 / (3r + 3)),
                (0 - r / (r + 3)) * (1 - 2 / (r + 3)),
                0,
            ],
            [
                (0 - r / (r + 1)) * (0 - 1 / (3r + 3)),
                (0 - r / (r + 3)) * (0 - 2 / (r + 3)),
                0,
            ],
            [
                (0 - r / (r + 1)) * (0 - 1 / (3r + 3)),
                (0 - r / (r + 3)) * (0 - 2 / (r + 3)),
                (0 - 0) * (1 - 1),
            ],
        ),
    )
    truth = sum(truthmat, dims = 2)[:]
    S = residuals(ft, type = "score")[:]
    @test all(isapprox.(S, truth))
    # assertions for testing breslow ties estimates under convergence for dat1
    if length(ft.P._LL) > 1
        @test isapprox(ft.P._B[1], 1.475285, atol = 0.000001)
        @test isapprox(ft.P._LL[[1, end]], [-4.56434819, -3.82474951], atol = 0.000001)
        @test isapprox(-ft.P._hess[1], 0.6341681, atol = 0.000001)
    end

    # TESTS: do cox models return the theoretic values appropriately for breslow and efron ties (following Terry Therneau's approach)
    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat1,
        keepx = true,
        keepy = true,
        ties = "breslow",
        maxiter = 0,
    )


    X = ft.P.X
    M = residuals(ft, type = "martingale")
    S = residuals(ft, type = "schoenfeld")[:]
    r = exp(ft.P._B[1])
    truthmat = [
        (1-r/(r+1))*(1-r/(3r+3)) 0 0
        (1-r/(r+1))*(0-r/(3r+3)) 0 0
        (1-r/(r+1))*(0-r/(3r+3)) (1-r/(r+3))*(1-2r/(r+3)) 0
        (0-r/(r+1))*(0-1/(3r+3)) (0-r/(r+3))*(1-2/(r+3)) 0
        (0-r/(r+1))*(0-1/(3r+3)) (0-r/(r+3))*(0-2/(r+3)) 0
        (0-r/(r+1))*(0-1/(3r+3)) (0-r/(r+3))*(0-2/(r+3)) (0-0)*(1-1)
    ]
    truth = sum(truthmat, dims = 1)[:]
    @test all(isapprox.(S, truth))

    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat1,
        keepx = true,
        keepy = true,
        ties = "breslow",
    )

    X = ft.P.X
    residuals(ft, type = "scaled_schoenfeld")[:]
    S = residuals(ft, type = "schoenfeld")[:]
    r = exp(ft.P._B[1])
    truthmat = [
        (1-r/(r+1))*(1-r/(3r+3)) 0 0
        (1-r/(r+1))*(0-r/(3r+3)) 0 0
        (1-r/(r+1))*(0-r/(3r+3)) (1-r/(r+3))*(1-2r/(r+3)) 0
        (0-r/(r+1))*(0-1/(3r+3)) (0-r/(r+3))*(1-2/(r+3)) 0
        (0-r/(r+1))*(0-1/(3r+3)) (0-r/(r+3))*(0-2/(r+3)) 0
        (0-r/(r+1))*(0-1/(3r+3)) (0-r/(r+3))*(0-2/(r+3)) (0-0)*(1-1)
    ]
    truth = sum(truthmat, dims = 1)[:]
    @test all(isapprox.(S, truth))

    ft = coxph(
        @formula(Surv(time, status) ~ x),
        dat3,
        wts = dat3.wt,
        keepx = true,
        keepy = true,
        ties = "efron",
    )
    rdfbetas = [
        0.6278889620495443,
        0.03530427452450842,
        0.12949839663825446,
        0.17266452885100594,
        -1.2767274051259652,
        -0.23852676492216027,
        0.2710638949411516,
        -0.19596100527839921,
        0.47479511834382615,
    ]
    @test isapprox(sqrt(vcov(ft, type = "robust")[1]), 1.1190551648004863)
    @test all(isapprox.(residuals(ft, type = "dfbetas"), rdfbetas))

    # TEST: can algorithms recover from an observation that doesn't contribute to the likelihood?
    # NOTE: no formal test here, this will error if incorrect
    id, int, outt, data = LSurvival.dgm(MersenneTwister(1232), 1000, 100; afun = LSurvival.int_0)
    data[:, 1] = round.(data[:, 1], digits = 3)
    d, X = data[:, 4], data[:, 1:3]
    wt = rand(length(d))
    #wt ./= (sum(wt) / length(wt))
    wt ./= wt
    xtab = (
        id = id,
        int = int,
        outt = outt,
        d = d,
        x = X[:, 1],
        z1 = X[:, 2],
        z2 = X[:, 3],
        wt = wt,
    )

    m = coxph(
        @formula(Surv(int, outt, d) ~ x + z1 + z2),
        xtab,
        wts = xtab.wt,
        id = ID.(xtab.id),
        ties = "breslow",
        keepx = true,
        keepy = true,
    )
    m2 = coxph(
        @formula(Surv(int, outt, d) ~ x + z1 + z2),
        xtab,
        wts = xtab.wt,
        id = ID.(xtab.id),
        ties = "efron",
        keepx = true,
        keepy = true,
    )

    #TEST: does RL get created appropriately and permanently?
    @test isnothing(m.RL)
    r = residuals(m, type = "dfbeta")
    @test !isnothing(m.RL)

    r2 = residuals(m2, type = "dfbeta")
    # Test: robust variance is different
    se = stderror(m)
    ser = stderror(m, type = "robust")
    @test se != ser

    mb = LSurvival.fit!(bootstrap(MersenneTwister(123), m), keepx = true, keepy = true)
    mb2 = LSurvival.fit!(bootstrap(MersenneTwister(123), m2), keepx = true, keepy = true)

    #TEST: does bootstrapping preserve model differences??

    @test coef(mb) !== coef(m)
    @test coef(mb2) !== coef(m2)
    @test mb2.ties == m2.ties
    @test mb.ties == m.ties

    #TEST: does seeding preserve bootstrap fidelity across different runs?
    mb2b = bootstrap(MersenneTwister(123), m2, 2)
    @test isapprox(coef(mb2), mb2b[1, :])

    #TEST: do bootstrap estimates vary across samples
    @test mb2b[1, :] != mb2b[2, :]

    #TEST: do confidence level estimates vary as expected across coverage values?
    @test confint(mb2, level = 0.95) != confint(mb2, level = 0.9)
    @test diff(confint(mb2, level = 0.95)[1, :]) > diff(confint(mb2, level = 0.90)[1, :])

    #TEST: do all systems give same value in seeding (NO)?
    #truemat = [1.75475  -0.414124  1.77553;1.90895  -0.116094  1.68474]
    #@test isapprox(mb2b, truemat, atol  = 0.00001) # comparison will fail on some 32 bit systems


    # TESTs: appropriate variance and robust variance with person-period data
    ft = coxph(@formula(Surv(time, status) ~ x), dat1, keepx = true)
    ft2 = coxph(
        @formula(Surv(enter, exit, status) ~ x),
        dat1clust,
        id = ID.(dat1clust.id),
        keepx = true,
    )
    ft2w = coxph(@formula(Surv(enter, exit, status) ~ x), dat1clust, keepx = true)
    
    #TEST: does model-based variance follow expected behavior, regardless of id argument?
    @test stderror(ft2) == stderror(ft)
    @test stderror(ft2) == stderror(ft2w)

    #TEST: does robust variance get created correctly in person period data when including id argument?
    @test stderror(ft2, type = "robust") == stderror(ft, type = "robust")
    #TEST: does robust variance fail to get created correctly in person period data when omitting id argument?
    @test stderror(ft2, type = "robust") != stderror(ft2w, type = "robust")

    # TEST: bootstrapping a survival response
    op = LSurvivalResp(zeros(100), rand(MersenneTwister(345), 100), ones(100))
    @test length(op.eventtimes) .> length(bootstrap(op)[2].eventtimes)
    
    # TEST: bootstrapping a competing survival response
    op = LSurvivalCompResp(zeros(100), rand(MersenneTwister(345), 100), rand(MersenneTwister(345), [0,1,2], 100))
    @test length(op.eventtimes) .> length(bootstrap(op)[2].eventtimes)

    # TEST: bootstrapping a cox model without a seed
    @test !bootstrap(ft2).fit
    
    # TEST: bootstrapping a kaplan meier without a seed
    # TODO: will this fail with weights?
    #km = kaplan_meier(zeros(length(dat1.time)), dat1.time, dat1.status)
    #@test bootstrap(km)

    # TEST: bootstrap errors
    ftf = coxph(@formula(Surv(time, status) ~ x), dat1, keepx = false)
    try bootstrap(ftf)
    catch e
      @test  typeof(e) == MethodError
    end
    ftf = coxph(@formula(Surv(time, status) ~ x), dat1, keepy = false)
    try bootstrap(ftf, 2)
    catch e
        @test  typeof(e) == String  # should this be converted to an error?
    end

    



end
