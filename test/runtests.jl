using Test
using LSurvival
using Random
#using DataFrames

@testset "LSurvival.jl" begin

    id, int, outt, data =
        LSurvival.dgm(MersenneTwister(1212), 20, 5; afun = LSurvival.int_0)

    d, X = data[:, 4], data[:, 1:3]
    weights = rand(length(d))

    # survival outcome:
    R = LSurvResp(int, outt, d, ID.(id))    # specification with ID only
    print(R)
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

    #data = DataFrame(data, [:x,:z1,:z2,:y])
    # f = @formula(y~z1+z2+x)
    # contrasts = nothing
    # M = M
    # modelframe(f, data, contrasts, M)


    LSurvival._fit!(M, start = [0.0, 0.0, 0.0])

    R = LSurvResp(int, outt, d)
    R = LSurvResp(outt, d) # set all to zero
    print(R)


    kaplan_meier(int, outt, d)
    #trivial case of non-competing events with late entry
    aalen_johansen(int, outt, d)


    z, x, t, d, event, wt = LSurvival.dgm_comprisk(MersenneTwister(1212), 100)
    enter = zeros(length(t))

    ajres = aalen_johansen(enter, t, event)
    aalen_johansen(enter, t, event, wts = wt)
    kms = kaplan_meier(enter, t, d, wts = wt)

    println(ajres)
    println(kms)

    X = hcat(z, x)
    d1 = d .* Int.(event .== 1)
    d2 = d .* Int.(event .== 2)

    ft2 = fit(PHModel, X, enter, t, (event .== 2), ties = "efron")
    ft1 = fit(PHModel, X, enter, t, (event .== 1), ties = "efron", verbose = true)
    coxph(X, enter, t, d2, ties = "efron")
    coeflist = [lnhr1, lnhr2]
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


    println(cires2b)

end
