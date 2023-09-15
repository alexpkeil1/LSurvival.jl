# to implement
# ID level robust variance estimators


using LSurvival, Random, Optim, BenchmarkTools, RCall, SpecialFunctions

######################################################################
# parametric survival model
######################################################################

#
SURVIVAL_LOG_LIKE=raw"""
$L(e_i, t_i) = f(t_i)^\delta_i/S(e_i) S(t_i)^(1-\delta_i)/S(e_i) $
for distribution function $f$ and survival function $S\equiv 
"""


"""
Weibull model for left truncated, right censored data
    
    $SURVIVAL_LOG_LIKE

ρ = .1    # linear predictor/shape parameter
d = 1
enter = 0
exit = 3.2
γ = 1   # transformation of scale parameter
wt = 1
loglik_weibull(enter, exit, d, wt, ρ, γ, α=1.)
"""
function loglik_weibull(enter, exit, d, wt, ρ, γ=1., α=1.)
    # ρ is linear predictor
    lambda = ρ*γ*exit^(γ-1.0)                            # hazard
    ll = -(ρ * exit)^γ
    ll += d == 1 ? log(ρ) + log(γ) + (γ-1)*log( ρ * exit)  : # extra contribution for events
                            0 # extra contribution for censored (cumulative conditional survival at censoring)
    ll -= enter > 0 ? (ρ * enter)^γ : 0 # (anti)-contribution for all in risk set (cumulative conditional survival at entry)
    ll *= wt
    ll
end
"""
ρ = .1    # linear predictor/shape parameter
d = 1
enter = 0
exit = 3.2
γ = 1   # transformation of scale parameter
wt = 1
loglik_exponential(enter, exit, d, wt, ρ, γ)
loglik_exponential(enter, exit, d, wt, ρ)
"""
loglik_exponential(enter, exit, d, wt, ρ, γ=1., α=1.) = loglik_weibull(enter, exit, d, wt, ρ, 1.0, 1.0)



raw"""
generalized gamma likelihood  - based on Lee and Wang book, pg. 188

Generalized gamma distribution function 
    $f(t) = \frac{\lambda}{\Gamma(\gamma)}(\lambda t)^{\gamma-1}\exp(-\lambda t) \hspace{20pt} t \geq 0, \hspace{10pt} \alpha, \gamma, \lambda > 0$
    $S(t) = 1 - I(\lambda t^\alpha, \beta$
with Gamma function 
$\Gamma(\gamma)
\begin{cases}
\int_0^\infty x^{\gamma-1}\exp(-x) dx\\ 
(\gamma - 1)! & \mbox{for integer } \gamma
\end{cases}$
and incomplete gamma function
$ \int_0^t u^{\gamma-1} \exp(-u)du/\Gamma(\gamma)$

Source: Lee and Wang, Klein and Moeschberger (Table 2.2)

Gamma distribution is the special case where $\alpha=1$

For events ($\delta_i$), right censoring ($1-\delta_i$) and left truncated observations ($\kappa_i$), in a person-period structure with no interval
 censoring

"""
function loglik_gengamma(enter, exit, d, wt, ρ, γ=1., α=1.)
    # ρ is linear predictor
    #;
    lambda = ρ*γ*exit^(γ-1.0)                            # hazard
    ll = 0.0
    # l = (abs(α)*(γ**γ)*(ρ**(α*γ))*(exit)**(α*γ-1)*exp(-γ*(ρ*ageout)**α))/SpecialFunctions.gamma(γ)
    
    ll += d == 1 ?  log(α) + γ*log(γ) + α*γ*log(ρ) + α*(γ-1)*log(exit) - γ*(ρ*exit)^α - SpecialFunctions.loggamma(γ) : # extra contribution for events
                            0 # extra contribution for censored (cumulative conditional survival at censoring)
    ll -= enter > 0 ?  : 0 # (anti)-contribution for all in risk set (cumulative conditional survival at entry)
    ll *= wt
    ll
end


function lpdf_gengamma(t,ρ, γ=1., α=1.)
    log(α) + γ*log(γ) + α*γ*log(ρ) + α*(γ-1)*log(t) - γ*(ρ*t)^α - SpecialFunctions.loggamma(γ)
end

function lsurv_gengamma(t,ρ, γ=1., α=1.)
    1 - gamma_inc(γ,x,IND=0)
end







#=
function expected_denj(_r, wts, caseidx, risksetidx, nties, j)
    # expected value denominator for all observations at a given time 
    _rcases = view(_r, caseidx)
    _rriskset = view(_r, risksetidx)
    _wtcases = view(wts, caseidx)
    _wtriskset = view(wts, risksetidx)
    #
    risksetrisk = sum(_wtriskset .* _rriskset)
    #
    effwts = LSurvival.efron_weights(nties)
    sw = sum(_wtcases)
    aw = sw / nties
    casesrisk = sum(_wtcases .* _rcases)
    dens = [risksetrisk - casesrisk * ew for ew in effwts]
    aw ./ dens # using Efron estimator
end
=#

#=
# commenting out to avoid issues with test coverage
    ######################################################################
    # fitting with optim (works, but more intensive than defaults)
    ######################################################################

    function fit!(
        m::PHModel;
        verbose::Bool=false,
        maxiter::Integer=500,
        atol::Float64=0.0,
        rtol::Float64=0.0,
        gtol::Float64=1e-8,
        start=nothing,
        keepx=false,
        keepy=false,
        bootstrap_sample=false,
        bootstrap_rng=MersenneTwister(),
        kwargs...
    )
        m = bootstrap_sample ? bootstrap(bootstrap_rng, m) : m
        start = isnothing(start) ? zeros(length(m.P._B)) : start
        if haskey(kwargs, :ties)
            m.ties = kwargs[:ties]
        end
        ne = length(m.R.eventtimes)
        risksetidxs, caseidxs =
            Array{Array{Int,1},1}(undef, ne), Array{Array{Int,1},1}(undef, ne)
        _sumwtriskset, _sumwtcase = zeros(Float64, ne), zeros(Float64, ne)
        @inbounds @simd for j = 1:ne
            _outj = m.R.eventtimes[j]
            fr = findall((m.R.enter .< _outj) .&& (m.R.exit .>= _outj))
            fc = findall(
                (m.R.y .> 0) .&& isapprox.(m.R.exit, _outj) .&& (m.R.enter .< _outj),
            )
            risksetidxs[j] = fr
            caseidxs[j] = fc
            _sumwtriskset[j] = sum(m.R.wts[fr])
            _sumwtcase[j] = sum(m.R.wts[fc])
        end
        # cox risk and set to zero were both in step cox - return them?
        # loop over event times
        #LSurvival._coxrisk!(m.P) # updates all elements of _r as exp(X*_B)
        #LSurvival._settozero!(m.P)
        #LSurvival._partial_LL!(m, risksetidxs, caseidxs, ne, den)

        function coxupdate!(
            F,
            G,
            H,
            beta,
            m;
            ne=ne,
            caseidxs=caseidxs,
            risksetidxs=risksetidxs
        )
            m.P._LL[1] = isnothing(F) ? m.P._LL[1] : F
            m.P._grad = isnothing(G) ? m.P._grad : G
            m.P._hess = isnothing(H) ? m.P._hess : H
            m.P._B = isnothing(beta) ? m.P._B : beta
            #
            LSurvival._update_PHParms!(m, ne, caseidxs, risksetidxs)
            # turn into a minimization problem
            F = -m.P._LL[1]
            m.P._grad .*= -1.0
            m.P._hess .*= -1.0
            F
        end

        fgh! = TwiceDifferentiable(
            Optim.only_fgh!((F, G, H, beta) -> coxupdate!(F, G, H, beta, m)),
            start,
        )
        opt = NewtonTrustRegion()
        #opt = IPNewton()
        #opt = Newton()
        res = optimize(
            fgh!,
            start,
            opt,
            Optim.Options(
                f_abstol=atol,
                f_reltol=rtol,
                g_tol=gtol,
                iterations=maxiter,
                store_trace=true,
            ),
        )
        verbose && println(res)

        m.fit = true
        m.P._grad .*= -1.0
        m.P._hess .*= -1.0
        m.P._LL = [-x.value for x in res.trace]
        basehaz!(m)
        m.P.X = keepx ? m.P.X : nothing
        m.R = keepy ? m.R : nothing
        m
    end

end # if false


if false
    id, int, outt, data =
        LSurvival.dgm(MersenneTwister(345), 100, 10; afun=LSurvival.int_0)
    data[:, 1] = round.(data[:, 1], digits=3)
    d, X = data[:, 4], data[:, 1:3]


    # not-yet-fit PH model object
    #m = PHModel(R, P, "breslow")
    #LSurvival._fit!(m, start = [0.0, 0.0, 0.0], keepx=true, keepy=true)
    #isfitted(m)
    R = LSurvivalResp(int, outt, d)
    P = PHParms(X)
    m = PHModel(R, P)  #default is "efron" method for ties
    @btime res = LSurvival._fit!(m, start=[0.0, 0.0, 0.0], keepx=true, keepy=true)

    R2 = LSurvivalResp(int, outt, d)
    P2 = PHParms(X)
    m2 = PHModel(R2, P2)  #default is "efron" method for ties
    @btime res2 = fit!(m2, start=[0.0, 0.0, 0.0], keepx=true, keepy=true)

    res
    res2

    argmax([res.P._LL[end], res2.P._LL[end]])

    res.P._LL[end] - res2.P._LL[end]

    # in progress functions
    # taken from GLM.jl/src/linpred.jl
=#

