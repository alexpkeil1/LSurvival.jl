# to implement
# ID level robust variance estimators


using LSurvival, Random, Optim
using Zygote # parametric survival models
using RCall
using BenchmarkTools
using LSurvival, Zygote, Random, StatsBase, Printf, Tables


using Symbolics, LinearAlgebra, LSurvival # 
using SpecialFunctions



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


#############################################
# testing Symbolics.jl for building gradient functions
#############################################
include("distributions.jl")
#= from distributions.jl
function lpdf_weibull(θ, ρ, t)
    z = (log(t) - α) / exp(ρ)
    ret = -ρ + z - exp(z) - log(t)
    ret
end

function lsurv_weibull(θ, ρ, t)
    z = (log(t) - α) / exp(ρ)
    -exp(z)
end

=#
# function for testing only

function lpdf_gengamma(α, ρ, κ, t)
    # l = -α*exp(-ρ) - exp(- α*exp(-ρ) + log(t)*exp(-ρ)) + log(t)*exp(-ρ) - ρ - log(t)
    z = (log(t) - α) * exp(-ρ)
    z*exp(κ) -exp(z)-ρ - log(t) - loggamma(exp(κ))
end



function lsurv_gengamma(α, ρ, κ, t)
    z = (log(t) - α) * exp(-ρ)
    p, _ = SpecialFunctions.gamma_inc_cf(κ, exp(z), 0)
    log1p(-p)
end


include("/Users/keilap/temp/gi.jl")
@variables α ρ κ t;
s = lpdf_gengamma(α,ρ,κ,t)
sgrad = Symbolics.gradient(s, [α,ρ,κ], simplify=true)
shess = Symbolics.hessian(s, [α,ρ,κ], simplify=true)

Symbolics.build_function(sgrad, α,ρ,κ, t, fname="survgrad_symbol")
Symbolics.build_function(shess, α,ρ,κ, t, fname="survhess_symbol")

"""
# confirmation function
θ = [-0.1, 0.2]
x = [1.0, -0.25]
ρ = -1.5
t = 0.8
α = dot(θ, x)
z = (log(t)-α)*exp(-ρ)

# finite difference
eps = 0.000001
[(lsurv_weibull(α, ρ, t)-lsurv_weibull(α-eps, ρ, t))/eps,
(lsurv_weibull(α, ρ, t)-lsurv_weibull(α, ρ-eps, t))/eps]

exp(z-ρ)


# analytic
dlsurv_weibull(α, ρ, t)
# symbolic
survgrad_symbol!(zeros(2), α, ρ, t, x)
"""
function survgrad_symbol!(ˍ₋out, α, ρ, t, x)
    begin
        @inbounds begin
            ˍ₋out[1] = (/)((exp)((/)((+)((*)(-1, α), (log)(t)), (exp)(ρ))), (exp)(ρ))
            ˍ₋out[2] = (/)((*)((+)((*)(-1, α), (log)(t)), (exp)((/)((+)((+)((*)(-1, α), (*)(ρ, (exp)(ρ))), (log)(t)), (exp)(ρ)))), (exp)((*)(2, ρ)))
        end
    end
    ˍ₋out
end



@variables α ρ t;
f = lpdf_test(α,ρ,t)
fgrad = Symbolics.gradient(f, [α,ρ], simplify=true)
fhess = Symbolics.hessian(f, [α,ρ], simplify=true)

Symbolics.build_function(fgrad, α,ρ, t, fname="pdfgrad_symbol")
Symbolics.build_function(fhess, α,ρ, t, fname="pdfhess_symbol")



@variables θ[1:2] ρ t[1:2] x[1:2,1:2];
f = lpdf_test(θ, ρ, t, x)
fgrad = Symbolics.gradient(f, [θ[1], θ[2], ρ], simplify=true)
fhess = Symbolics.hessian(f, [θ[1], θ[2], ρ], simplify=true)

Symbolics.build_function(fgrad, θ, ρ, t, fname="pdfgrad_symbol")
Symbolics.build_function(fhess, θ, ρ, t, fname="pdfhess_symbol")

@variables θ[1:2] ρ t[1:2] x[1:2,1:2];
size(x)
s = lsurv_test(θ, ρ, t, x)
sgrad = Symbolics.gradient(s, [θ[1], θ[2], ρ], simplify=true)
shess = Symbolics.hessian(s, [θ[1], θ[2], ρ], simplify=true)

Symbolics.build_function(sgrad, θ, ρ, t, fname="survgrad_symbol")
Symbolics.build_function(shess, θ, ρ, t, fname="survhess_symbol")



"""
# confirmation function
θ = [-0.1, 0.2]
x = [1.0  -0.25; 1 .25]
ρ = 1.5
t = [0.8, .5]
#α = dot(θ, x)


LSurvival.lpdf_gradient(LSurvival.Weibull(), vcat(θ, ρ), t[1], x[1,:]) + LSurvival.lpdf_gradient(LSurvival.Weibull(), vcat(θ, ρ), t[2], x[2,:])
pdfgrad_symbol!(zeros(3), θ, ρ, t, x)
"""
function pdfgrad_symbol!(ˍ₋out, θ, ρ, t, x)
    begin
        @inbounds begin
            ˍ₋out[1] = (/)((+)((+)((+)((*)(-1.0, (getindex)(x, 1, 1)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (getindex)(x, 1, 1))), (*)(-1.0, (getindex)(x, 2, 1))), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (getindex)(x, 2, 1))), (exp)(ρ))
            ˍ₋out[2] = (/)((+)((+)((+)((*)(-1.0, (getindex)(x, 1, 2)), (*)(-1.0, (getindex)(x, 2, 2))), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (getindex)(x, 1, 2))), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (getindex)(x, 2, 2))), (exp)(ρ))
            ˍ₋out[3] = (/)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((*)(-2.0, (exp)((*)(2, ρ))), (*)((exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (log)((getindex)(t, 1)))), (*)((exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (log)((getindex)(t, 2)))), (*)((*)((exp)(ρ), (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)((exp)(ρ), (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1.0, (exp)(ρ)), (log)((getindex)(t, 1)))), (*)((*)(-1.0, (exp)(ρ)), (log)((getindex)(t, 2)))), (*)((*)((exp)(ρ), (getindex)(θ, 2)), (getindex)(x, 1, 2))), (*)((*)((exp)(ρ), (getindex)(θ, 2)), (getindex)(x, 2, 2))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(θ, 2)), (getindex)(x, 1, 2))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(θ, 2)), (getindex)(x, 2, 2))), (exp)((*)(2, ρ)))
        end
    end
    ˍ₋out
end

"""
# confirmation function
θ = [-0.1, 0.2]
x = [1.0  -0.25; 1 .25]
ρ = 1.5
t = [0.8, .5]
#α = dot(θ, x)

LSurvival.lsurv_gradient(LSurvival.Weibull(), vcat(θ, ρ), t[1], x[1,:]) + LSurvival.lsurv_gradient(LSurvival.Weibull(), vcat(θ, ρ), t[2], x[2,:])
survgrad_symbol!(zeros(3), θ, ρ, t, x)
"""
function survgrad_symbol!(ˍ₋out, θ, ρ, t, x)
    begin
        @inbounds begin
            ˍ₋out[1] = (/)((+)((*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (getindex)(x, 1, 1)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (getindex)(x, 2, 1))), (exp)(ρ))
            ˍ₋out[2] = (/)((+)((*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (getindex)(x, 1, 2)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (getindex)(x, 2, 2))), (exp)(ρ))
            ˍ₋out[3] = (/)((+)((+)((+)((+)((+)((*)((exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (log)((getindex)(t, 1))), (*)((exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (log)((getindex)(t, 2)))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(θ, 2)), (getindex)(x, 1, 2))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(θ, 2)), (getindex)(x, 2, 2))), (exp)((*)(2, ρ)))
        end
    end
    ˍ₋out
end




function fun()
    z= (log(t) - α) / exp(ρ)
    (α*exp(ρ) + ((ρ*exp(ρ) + exp(ρ)) / exp(ρ) - ((ρ*exp(ρ) + log(t) - α) / (exp(ρ)^2))*exp(ρ))*log(t)*exp((ρ*exp(ρ) + log(t) - α) / exp(ρ)) - 2.0exp(2ρ) - log(t)*exp(ρ) - α*((ρ*exp(ρ) + exp(ρ)) / exp(ρ) - ((ρ*exp(ρ) + log(t) - α) / (exp(ρ)^2))*exp(ρ))*exp((ρ*exp(ρ) + log(t) - α) / exp(ρ))) / exp(2ρ) - 2((α*exp(ρ) + log(t)*exp((ρ*exp(ρ) + log(t) - α) / exp(ρ)) - exp(2ρ) - log(t)*exp(ρ) - α*exp((ρ + z))) / exp(ρ)^2)
end
fun()


"""
θ = [-0.1, 0.2]
x = [1.0  -0.25; 1 .25]
ρ = 1.5
t = [0.8, .5]
#α = dot(θ, x)


LSurvival.lpdf_hessian(LSurvival.Weibull(), vcat(θ, ρ), t[1], x[1,:]) + LSurvival.lpdf_hessian(LSurvival.Weibull(), vcat(θ, ρ), t[2], x[2,:])
pdfhess_symbol!(zeros(3,3), θ, ρ, t, x)
"""
function pdfhess_symbol!(ˍ₋out, θ, ρ, t, x)
    begin
        @inbounds begin
            ˍ₋out[1] = (/)((+)((/)((*)((*)(-1, (^)((getindex)(x, 1, 1), 2)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (exp)(ρ)), (/)((*)((*)(-1, (^)((getindex)(x, 2, 1), 2)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (exp)(ρ))), (exp)(ρ))
            ˍ₋out[2] = (/)((+)((/)((*)((*)((*)(-1, (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(x, 1, 1)), (getindex)(x, 1, 2)), (exp)(ρ)), (/)((*)((*)((*)(-1, (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(x, 2, 1)), (getindex)(x, 2, 2)), (exp)(ρ))), (exp)(ρ))
            ˍ₋out[3] = (+)((/)((+)((*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(x, 1, 1)), (*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(x, 2, 1))), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(-1.0, (getindex)(x, 1, 1)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (getindex)(x, 1, 1))), (*)(-1.0, (getindex)(x, 2, 1))), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (getindex)(x, 2, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)))
            ˍ₋out[4] = (/)((+)((/)((*)((*)((*)(-1, (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(x, 1, 1)), (getindex)(x, 1, 2)), (exp)(ρ)), (/)((*)((*)((*)(-1, (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(x, 2, 1)), (getindex)(x, 2, 2)), (exp)(ρ))), (exp)(ρ))
            ˍ₋out[5] = (/)((+)((/)((*)((*)(-1, (^)((getindex)(x, 1, 2), 2)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (exp)(ρ)), (/)((*)((*)(-1, (^)((getindex)(x, 2, 2), 2)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (exp)(ρ))), (exp)(ρ))
            ˍ₋out[6] = (+)((/)((+)((*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(x, 1, 2)), (*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(x, 2, 2))), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(-1.0, (getindex)(x, 1, 2)), (*)(-1.0, (getindex)(x, 2, 2))), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (getindex)(x, 1, 2))), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (getindex)(x, 2, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)))
            ˍ₋out[7] = (+)((/)((+)((*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(x, 1, 1)), (*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(x, 2, 1))), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(-1.0, (getindex)(x, 1, 1)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (getindex)(x, 1, 1))), (*)(-1.0, (getindex)(x, 2, 1))), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (getindex)(x, 2, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)))
            ˍ₋out[8] = (+)((/)((+)((*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(x, 1, 2)), (*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(x, 2, 2))), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(-1.0, (getindex)(x, 1, 2)), (*)(-1.0, (getindex)(x, 2, 2))), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (getindex)(x, 1, 2))), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (getindex)(x, 2, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)))
            ˍ₋out[9] = (+)((/)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((*)(-4.0, (exp)((*)(2, ρ))), (*)((*)((exp)(ρ), (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)((exp)(ρ), (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1.0, (exp)(ρ)), (log)((getindex)(t, 1)))), (*)((*)(-1.0, (exp)(ρ)), (log)((getindex)(t, 2)))), (*)((*)((+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (log)((getindex)(t, 1)))), (*)((*)((+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (log)((getindex)(t, 2)))), (*)((*)((exp)(ρ), (getindex)(θ, 2)), (getindex)(x, 1, 2))), (*)((*)((exp)(ρ), (getindex)(θ, 2)), (getindex)(x, 2, 2))), (*)((*)((*)((*)(-1.0, (+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)((*)((*)(-1.0, (+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(θ, 2)), (getindex)(x, 1, 2))), (*)((*)((*)((*)(-1.0, (+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)((*)((*)(-1.0, (+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(θ, 2)), (getindex)(x, 2, 2))), (exp)((*)(2, ρ))), (*)((*)(-2, (/)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((+)((*)(-2.0, (exp)((*)(2, ρ))), (*)((exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (log)((getindex)(t, 1)))), (*)((exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (log)((getindex)(t, 2)))), (*)((*)((exp)(ρ), (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)((exp)(ρ), (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1.0, (exp)(ρ)), (log)((getindex)(t, 1)))), (*)((*)(-1.0, (exp)(ρ)), (log)((getindex)(t, 2)))), (*)((*)((exp)(ρ), (getindex)(θ, 2)), (getindex)(x, 1, 2))), (*)((*)((exp)(ρ), (getindex)(θ, 2)), (getindex)(x, 2, 2))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(θ, 2)), (getindex)(x, 1, 2))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(θ, 2)), (getindex)(x, 2, 2))), (^)((exp)((*)(2, ρ)), 2))), (exp)((*)(2, ρ))))
        end
    end
    ˍ₋out
end

"""
# confirmation function
θ = [-0.1, 0.2]
x = [1.0  -0.25; 1 .25]
ρ = 1.5
t = [0.8, .5]
#α = dot(θ, x)

ga = LSurvival.lsurv_gradient(LSurvival.Weibull(), vcat(θ, ρ), t[1], x[1,:]) + LSurvival.lsurv_gradient(LSurvival.Weibull(), vcat(θ, ρ), t[2], x[2,:]) + LSurvival.lpdf_gradient(LSurvival.Weibull(), vcat(θ, ρ), t[1], x[1,:]) + LSurvival.lpdf_gradient(LSurvival.Weibull(), vcat(θ, ρ), t[2], x[2,:])
gb = survgrad_symbol!(zeros(3), θ, ρ, t, x) + pdfgrad_symbol!(zeros(3), θ, ρ, t, x)

ha = LSurvival.lsurv_hessian(LSurvival.Weibull(), vcat(θ, ρ), t[1], x[1,:]) + LSurvival.lsurv_hessian(LSurvival.Weibull(), vcat(θ, ρ), t[2], x[2,:]) + LSurvival.lpdf_hessian(LSurvival.Weibull(), vcat(θ, ρ), t[1], x[1,:]) + LSurvival.lpdf_hessian(LSurvival.Weibull(), vcat(θ, ρ), t[2], x[2,:])
hb = survhess_symbol!(zeros(3,3), θ, ρ, t, x) + pdfhess_symbol!(zeros(3,3), θ, ρ, t, x)

# newton raphson step
inv(ha) * ga
inv(hb) * gb

"""
function survhess_symbol!(ˍ₋out, θ, ρ, t, x)
    begin
        @inbounds begin
            ˍ₋out[1] = (/)((+)((/)((*)((*)(-1, (^)((getindex)(x, 1, 1), 2)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (exp)(ρ)), (/)((*)((*)(-1, (^)((getindex)(x, 2, 1), 2)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (exp)(ρ))), (exp)(ρ))
            ˍ₋out[2] = (/)((+)((/)((*)((*)((*)(-1, (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(x, 1, 1)), (getindex)(x, 1, 2)), (exp)(ρ)), (/)((*)((*)((*)(-1, (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(x, 2, 1)), (getindex)(x, 2, 2)), (exp)(ρ))), (exp)(ρ))
            ˍ₋out[3] = (+)((/)((+)((*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(x, 1, 1)), (*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(x, 2, 1))), (exp)(ρ)), (*)((*)(-1, (/)((+)((*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (getindex)(x, 1, 1)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (getindex)(x, 2, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)))
            ˍ₋out[4] = (/)((+)((/)((*)((*)((*)(-1, (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(x, 1, 1)), (getindex)(x, 1, 2)), (exp)(ρ)), (/)((*)((*)((*)(-1, (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(x, 2, 1)), (getindex)(x, 2, 2)), (exp)(ρ))), (exp)(ρ))
            ˍ₋out[5] = (/)((+)((/)((*)((*)(-1, (^)((getindex)(x, 1, 2), 2)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (exp)(ρ)), (/)((*)((*)(-1, (^)((getindex)(x, 2, 2), 2)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (exp)(ρ))), (exp)(ρ))
            ˍ₋out[6] = (+)((/)((+)((*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(x, 1, 2)), (*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(x, 2, 2))), (exp)(ρ)), (*)((*)(-1, (/)((+)((*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (getindex)(x, 1, 2)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (getindex)(x, 2, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)))
            ˍ₋out[7] = (+)((/)((+)((*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(x, 1, 1)), (*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(x, 2, 1))), (exp)(ρ)), (*)((*)(-1, (/)((+)((*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (getindex)(x, 1, 1)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (getindex)(x, 2, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)))
            ˍ₋out[8] = (+)((/)((+)((*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(x, 1, 2)), (*)((*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(x, 2, 2))), (exp)(ρ)), (*)((*)(-1, (/)((+)((*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (getindex)(x, 1, 2)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1)), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (getindex)(x, 2, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)))
            ˍ₋out[9] = (+)((/)((+)((+)((+)((+)((+)((*)((*)((+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (log)((getindex)(t, 1))), (*)((*)((+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (log)((getindex)(t, 2)))), (*)((*)((*)((*)(-1.0, (+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)((*)((*)(-1.0, (+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)((*)((*)(-1.0, (+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(θ, 2)), (getindex)(x, 1, 2))), (*)((*)((*)((*)(-1.0, (+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(θ, 2)), (getindex)(x, 2, 2))), (exp)((*)(2, ρ))), (*)((*)(-2, (/)((+)((+)((+)((+)((+)((*)((exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ))), (log)((getindex)(t, 1))), (*)((exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ))), (log)((getindex)(t, 2)))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 1, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 1, 2))), (log)((getindex)(t, 1))), (exp)(ρ)))), (getindex)(θ, 2)), (getindex)(x, 1, 2))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(θ, 1)), (getindex)(x, 2, 1))), (*)((*)(-1, (getindex)(θ, 2)), (getindex)(x, 2, 2))), (log)((getindex)(t, 2))), (exp)(ρ)))), (getindex)(θ, 2)), (getindex)(x, 2, 2))), (^)((exp)((*)(2, ρ)), 2))), (exp)((*)(2, ρ))))
        end
    end
    ˍ₋out
end


#=
function testblock()
    # testing gradient function
    θ = [-2, 1.2]
    x = [1, 0.1]
    ρ = -0.5
    t = 3.0
    α = dot(θ, x)
    #dlpdf_weibull(dot(θ,x), ρ, t)
    z = (log(t) - α) / exp(ρ)
    truth = [((exp(z) - 1) * x[1]) / exp(ρ),  # 222
        ((exp(z) - 1) * x[2]) / exp(ρ),  # 222*0.1
        (log(t) * exp((ρ * exp(ρ) + log(t) - x[1] * θ[1] - x[2] * θ[2]) / exp(ρ)) + exp(ρ) * x[1] * θ[1] + exp(ρ) * x[2] * θ[2] - exp(2ρ) - log(t) * exp(ρ) - exp((ρ * exp(ρ) + log(t) - x[1] * θ[1] - x[2] * θ[2]) / exp(ρ)) * x[1] * θ[1] - exp((ρ * exp(ρ) + log(t) - x[1] * θ[1] - x[2] * θ[2]) / exp(ρ)) * x[2] * θ[2]) / exp(2ρ)]

    all(isapprox.(dlpdf_regweibull(θ, ρ, t, x), truth))



    # testing hessian function

    ddlpdf_weibull(dot([-2, 1.2], [1, 1]), -0.5, 3)

end
=#







