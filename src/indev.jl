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
function lpdf_weibull(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    ret = -ρ + z - exp(z) - log(t)
    ret
end

function lsurv_weibull(α, ρ, t)
    z = (log(t) - α) / exp(ρ)
    -exp(z)
end

=#
# function for testing only

function lpdf_lognormalint(θ, ρ, t, x)
    z = (log(t) - dot(θ,x)) / exp(ρ)
    ret = -log(sqrt(2pi)) - log(t) - ρ
    ret += -z * z / 2.0
    ret
end

function lsurv_lognormalint(θ, ρ, t, x)
    z = (log(t) - dot(θ,x)) / exp(ρ)
    ret = log(1 - cdfnorm(z))
    ret
end


@variables α ρ t;
s = lsurv_lognormal(α, ρ, t)
sgrad = Symbolics.gradient(s, [α, ρ], simplify=true)
shess = Symbolics.hessian(s, [α, ρ], simplify=true)

Symbolics.build_function(sgrad, α, ρ, t, fname="survgrad_symbol")
Symbolics.build_function(shess, α, ρ, t, fname="survhess_symbol")


@variables α ρ t;
f = lpdf_lognormal(α, ρ, t)
fgrad = Symbolics.gradient(f, [α, ρ], simplify=true)
fhess = Symbolics.hessian(f, [α, ρ], simplify=true)

Symbolics.build_function(fgrad, α, ρ, t, fname="pdfgrad_symbol")
Symbolics.build_function(fhess, α, ρ, t, fname="pdfhess_symbol")


#=
 (0.39894228040143265exp((-0.5((log(t) - α)^2)) / (exp(ρ)^2))) / ((0.5 - 0.5*erf((0.7071067811865475(log(t) - α)) / exp(ρ)))*exp(ρ))
 (0.39894228040143265(log(t) - α)*exp((ρ*(exp(ρ)^2) + 1.0α*log(t) - 0.5(α^2) - 0.5(log(t)^2)) / (exp(ρ)^2))) / ((0.5 - 0.5*erf((0.7071067811865475(log(t) - α)) / exp(ρ)))*exp(2ρ))
=#



#=
=#

#=
-sqrt(2/pi) = -0.7978845608028653
1/sqrt(2) = 0.7071067811865475
1/sqrt(2pi) = 0.39894228040143265
1/sqrt(pi) = 0.5641895835477563
qstdnorm(p) = sqrt(2) * erfinv(2.0 * p - 1.0)
cdfnorm(z) = 0.5 * (1 + erf(z / sqrt(2)))
1-cdfnorm(z) = (0.5 - 0.5*erf((inv(sqrt(2))*z)))
=#



"""
# confirmation function
θ = [-2, 1.2]
x = [1, 0.1]
ρ = -0.5
t = 3.0
α = dot(θ, x)


dlpdf_lognormal(α, ρ, t)
pdfgrad_symbol!(zeros(2), α, ρ, t)
"""
function pdfgrad_symbol!(ˍ₋out, α, ρ, t)
    begin
        @inbounds begin
            ˍ₋out[1] = (/)((+)((*)(-1, α), (log)(t)), (exp)((*)(2, ρ)))
            ˍ₋out[2] = (+)(-1, (/)((^)((+)((*)(-1, α), (log)(t)), 2), (exp)((*)(2, ρ))))
        end
    end
    ˍ₋out
end

"""
# confirmation function
θ = [-2, 1.2]
x = [1, 0.1]
ρ = -0.5
t = 3.0
α = dot(θ, x)

pdfhess_symbol!(zeros(2,2), α, ρ, t)
ddlpdf_lognormal(α, ρ, t)

"""
function pdfhess_symbol!(ˍ₋out, α, ρ, t)
    begin
        @inbounds begin
            ˍ₋out[1] = (/)(-1, (exp)((*)(2, ρ)))
            ˍ₋out[2] = (*)((*)(-2, (/)((+)((*)(-1, α), (log)(t)), (^)((exp)((*)(2, ρ)), 2))), (exp)((*)(2, ρ)))
            ˍ₋out[3] = (*)((*)(-2, (/)((+)((*)(-1, α), (log)(t)), (^)((exp)((*)(2, ρ)), 2))), (exp)((*)(2, ρ)))
            ˍ₋out[4] = (+)((/)((*)((*)(2, (^)((+)((*)(-1, α), (log)(t)), 2)), (exp)((*)(2, ρ))), (exp)((*)(4, ρ))), (*)((*)(-4, (/)((*)((^)((+)((*)(-1, α), (log)(t)), 2), (exp)((*)(2, ρ))), (^)((exp)((*)(4, ρ)), 2))), (exp)((*)(4, ρ))))
        end
    end
    ˍ₋out
end




@variables θ[1:2] ρ t x[1:2];
sint = lsurv_lognormalint(θ, ρ, t, x)
sgradint = Symbolics.gradient(sint, [θ[1], θ[2], ρ], simplify=true)
shessint = Symbolics.hessian(sint, [θ[1], θ[2], ρ], simplify=true)

Symbolics.build_function(sgradint, α, ρ, t, x, fname="survintgrad_symbol")
Symbolics.build_function(shessint, α, ρ, t, x, fname="survinthess_symbol")


# built functions
"""
θ = [-2, 1.2]
x = [1, 0.1]
ρ = -0.5
t = 3.0
α = dot(θ, x)

dlsurv_lognormal(α, ρ, t)
dlsurv_reglognormal(θ, ρ, t, x)
survintgrad_symbol!(zeros(3), θ, ρ, t, x)

"""
function survintgrad_symbol!(ˍ₋out, α, ρ, t, x)
    begin
        @inbounds begin
            ˍ₋out[1] = (/)((*)((*)(0.39894228040143265, (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 1)), (*)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), (exp)(ρ)))
                      ˍ₋out[2] = (/)((*)((*)(0.39894228040143265, (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 2)), (*)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), (exp)(ρ)))
                      ˍ₋out[3] = (/)((*)((*)(0.39894228040143265, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)((/)((+)((+)((+)((+)((+)((+)((*)(-0.4999999999999999, (^)((log)(t), 2)), (*)(ρ, (^)((exp)(ρ), 2))), (*)((*)(-0.4999999999999999, (^)((getindex)(x, 1), 2)), (^)((getindex)(θ, 1), 2))), (*)((*)(-0.4999999999999999, (^)((getindex)(x, 2), 2)), (^)((getindex)(θ, 2), 2))), (*)((*)((*)(0.9999999999999998, (log)(t)), (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)((*)(0.9999999999999998, (log)(t)), (getindex)(x, 2)), (getindex)(θ, 2))), (*)((*)((*)((*)(-0.9999999999999998, (getindex)(x, 1)), (getindex)(x, 2)), (getindex)(θ, 1)), (getindex)(θ, 2))), (^)((exp)(ρ), 2)))), (*)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), (exp)((*)(2, ρ))))
            nothing
        end
    end
    ˍ₋out
end

# built functions
"""
# confirmation function
θ = [-2, 1.2]
x = [1, 0.1]
ρ = -0.5
t = 3.0
α = dot(θ, x)

ddlsurv_lognormal(α, ρ, t)
ddlsurv_reglognormal(θ, ρ, t, x)
survinthess_symbol!(zeros(3,3), θ, ρ, t, x)

"""
function survinthess_symbol!(ˍ₋out, α, ρ, t, x)
    begin
         @inbounds begin
                  #= /Users/keilap/.julia/packages/SymbolicUtils/YVse6/src/code.jl:418 =#
                  ˍ₋out[1] = (+)((/)((*)((*)((*)(0.39894228040143254, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (^)((getindex)(x, 1), 2)), (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (*)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), (^)((exp)(ρ), 3))), (*)((*)((*)(-0.39894228040143265, (/)((*)((*)(0.39894228040143265, (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 1)), (*)((^)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), 2), (^)((exp)(ρ), 2)))), (exp)((*)(-1, (^)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)), 2)))), (getindex)(x, 1)))
                  ˍ₋out[2] = (+)((/)((*)((*)((*)((*)(0.39894228040143254, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 1)), (getindex)(x, 2)), (*)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), (^)((exp)(ρ), 3))), (*)((*)((*)(-0.39894228040143265, (/)((*)((*)(0.39894228040143265, (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 1)), (*)((^)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), 2), (^)((exp)(ρ), 2)))), (exp)((*)(-1, (^)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)), 2)))), (getindex)(x, 2)))
                  ˍ₋out[3] = (+)((/)((*)((*)((*)((*)(-0.7978845608028653, (/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 4))), (exp)(ρ)), (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 1)), (+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)))))), (*)((*)(-1, (+)((*)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), (exp)(ρ)), (*)((*)((*)(0.5641895835477563, (/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (^)((exp)(ρ), 2))), (^)((exp)(ρ), 2)), (exp)((*)(-1, (^)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)), 2)))))), (/)((*)((*)(0.39894228040143265, (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 1)), (*)((^)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), 2), (^)((exp)(ρ), 2)))))
                  ˍ₋out[4] = (+)((/)((*)((*)((*)((*)(0.39894228040143254, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 1)), (getindex)(x, 2)), (*)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), (^)((exp)(ρ), 3))), (*)((*)((*)(-0.39894228040143265, (/)((*)((*)(0.39894228040143265, (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 1)), (*)((^)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), 2), (^)((exp)(ρ), 2)))), (exp)((*)(-1, (^)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)), 2)))), (getindex)(x, 2)))
                  ˍ₋out[5] = (+)((/)((*)((*)((*)(0.39894228040143254, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (^)((getindex)(x, 2), 2)), (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (*)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), (^)((exp)(ρ), 3))), (*)((*)((*)(-0.39894228040143265, (/)((*)((*)(0.39894228040143265, (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 2)), (*)((^)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), 2), (^)((exp)(ρ), 2)))), (exp)((*)(-1, (^)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)), 2)))), (getindex)(x, 2)))
                  ˍ₋out[6] = (+)((/)((*)((*)((*)((*)(-0.7978845608028653, (/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 4))), (exp)(ρ)), (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 2)), (+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)))))), (*)((*)(-1, (+)((*)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), (exp)(ρ)), (*)((*)((*)(0.5641895835477563, (/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (^)((exp)(ρ), 2))), (^)((exp)(ρ), 2)), (exp)((*)(-1, (^)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)), 2)))))), (/)((*)((*)(0.39894228040143265, (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 2)), (*)((^)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), 2), (^)((exp)(ρ), 2)))))
                  ˍ₋out[7] = (+)((/)((*)((*)((*)((*)(-0.7978845608028653, (/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 4))), (exp)(ρ)), (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 1)), (+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)))))), (*)((*)(-1, (+)((*)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), (exp)(ρ)), (*)((*)((*)(0.5641895835477563, (/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (^)((exp)(ρ), 2))), (^)((exp)(ρ), 2)), (exp)((*)(-1, (^)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)), 2)))))), (/)((*)((*)(0.39894228040143265, (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 1)), (*)((^)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), 2), (^)((exp)(ρ), 2)))))
                  ˍ₋out[8] = (+)((/)((*)((*)((*)((*)(-0.7978845608028653, (/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 4))), (exp)(ρ)), (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 2)), (+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)))))), (*)((*)(-1, (+)((*)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), (exp)(ρ)), (*)((*)((*)(0.5641895835477563, (/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (^)((exp)(ρ), 2))), (^)((exp)(ρ), 2)), (exp)((*)(-1, (^)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)), 2)))))), (/)((*)((*)(0.39894228040143265, (exp)((/)((*)(-0.4999999999999999, (^)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), 2)), (^)((exp)(ρ), 2)))), (getindex)(x, 2)), (*)((^)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), 2), (^)((exp)(ρ), 2)))))
                  ˍ₋out[9] = (+)((/)((*)((*)((*)(0.39894228040143265, (+)((/)((+)((^)((exp)(ρ), 2), (*)((*)(2, ρ), (^)((exp)(ρ), 2))), (^)((exp)(ρ), 2)), (*)((*)(-2, (/)((+)((+)((+)((+)((+)((+)((*)(-0.4999999999999999, (^)((log)(t), 2)), (*)(ρ, (^)((exp)(ρ), 2))), (*)((*)(-0.4999999999999999, (^)((getindex)(x, 1), 2)), (^)((getindex)(θ, 1), 2))), (*)((*)(-0.4999999999999999, (^)((getindex)(x, 2), 2)), (^)((getindex)(θ, 2), 2))), (*)((*)((*)(0.9999999999999998, (log)(t)), (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)((*)(0.9999999999999998, (log)(t)), (getindex)(x, 2)), (getindex)(θ, 2))), (*)((*)((*)((*)(-0.9999999999999998, (getindex)(x, 1)), (getindex)(x, 2)), (getindex)(θ, 1)), (getindex)(θ, 2))), (^)((exp)(ρ), 4))), (^)((exp)(ρ), 2)))), (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)((/)((+)((+)((+)((+)((+)((+)((*)(-0.4999999999999999, (^)((log)(t), 2)), (*)(ρ, (^)((exp)(ρ), 2))), (*)((*)(-0.4999999999999999, (^)((getindex)(x, 1), 2)), (^)((getindex)(θ, 1), 2))), (*)((*)(-0.4999999999999999, (^)((getindex)(x, 2), 2)), (^)((getindex)(θ, 2), 2))), (*)((*)((*)(0.9999999999999998, (log)(t)), (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)((*)(0.9999999999999998, (log)(t)), (getindex)(x, 2)), (getindex)(θ, 2))), (*)((*)((*)((*)(-0.9999999999999998, (getindex)(x, 1)), (getindex)(x, 2)), (getindex)(θ, 1)), (getindex)(θ, 2))), (^)((exp)(ρ), 2)))), (*)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), (exp)((*)(2, ρ)))), (*)((*)(-1, (+)((*)((*)(2, (+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)))))), (exp)((*)(2, ρ))), (*)((*)((*)((*)(0.5641895835477563, (/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (^)((exp)(ρ), 2))), (exp)(ρ)), (exp)((*)(-1, (^)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ)), 2)))), (exp)((*)(2, ρ))))), (/)((*)((*)(0.39894228040143265, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)((/)((+)((+)((+)((+)((+)((+)((*)(-0.4999999999999999, (^)((log)(t), 2)), (*)(ρ, (^)((exp)(ρ), 2))), (*)((*)(-0.4999999999999999, (^)((getindex)(x, 1), 2)), (^)((getindex)(θ, 1), 2))), (*)((*)(-0.4999999999999999, (^)((getindex)(x, 2), 2)), (^)((getindex)(θ, 2), 2))), (*)((*)((*)(0.9999999999999998, (log)(t)), (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)((*)(0.9999999999999998, (log)(t)), (getindex)(x, 2)), (getindex)(θ, 2))), (*)((*)((*)((*)(-0.9999999999999998, (getindex)(x, 1)), (getindex)(x, 2)), (getindex)(θ, 1)), (getindex)(θ, 2))), (^)((exp)(ρ), 2)))), (*)((^)((+)(0.5, (*)(-0.5, (SpecialFunctions.erf)((/)((*)(0.7071067811865475, (+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t))), (exp)(ρ))))), 2), (^)((exp)((*)(2, ρ)), 2)))))
        end
    end
    ˍ₋out
end



@variables θ[1:2] ρ t x[1:2];
fint = lpdf_weibullint(θ, ρ, t, x)
fgradint = Symbolics.gradient(fint, [θ[1], θ[2], ρ], simplify=true)
fhessint = Symbolics.hessian(fint, [θ[1], θ[2], ρ], simplify=true)

Symbolics.build_function(fgradint, α, ρ, t, x, fname="pdfintgrad_symbol")
Symbolics.build_function(fhessint, α, ρ, t, x, fname="pdfinthess_symbol")




# built functions
"""
θ = [-2, 1.2]
x = [1, 0.1]
ρ = -0.5
t = 3.0
α = dot(θ, x)

dlpdf_weibull(α, ρ, t)
dlpdf_regweibull(θ, ρ, t, x)
pdfintgrad_symbol!(zeros(3), θ, ρ, t, x)

"""
function pdfintgrad_symbol!(ˍ₋out, α, ρ, t, x)
    begin
        @inbounds begin
            ˍ₋out[1] = (/)((+)((*)(-1.0, (getindex)(x, 1)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ))), (getindex)(x, 1))), (exp)(ρ))
            ˍ₋out[2] = (/)((+)((*)(-1.0, (getindex)(x, 2)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ))), (getindex)(x, 2))), (exp)(ρ))
            ˍ₋out[3] = (/)((+)((+)((+)((+)((+)((+)((*)(-1.0, (exp)((*)(2, ρ))), (*)((log)(t), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ))))), (*)((*)(-1.0, (log)(t)), (exp)(ρ))), (*)((*)((exp)(ρ), (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)((exp)(ρ), (getindex)(x, 2)), (getindex)(θ, 2))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (getindex)(x, 2)), (getindex)(θ, 2))), (exp)((*)(2, ρ)))
        end
    end
    ˍ₋out
end

# built functions
"""
# confirmation function
θ = [-2, 1.2]
x = [1, 0.1]
ρ = -0.5
t = 3.0
α = dot(θ, x)

ddlpdf_weibull(α, ρ, t)
ddlpdf_regweibull(θ, ρ, t, x)
pdfinthess_symbol!(zeros(3,3), θ, ρ, t, x)

"""
function pdfinthess_symbol!(ˍ₋out, α, ρ, t, x)
    begin
        @inbounds begin
            ˍ₋out[1] = (/)((*)((*)(-1, (^)((getindex)(x, 1), 2)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (^)((exp)(ρ), 2))
            ˍ₋out[2] = (/)((*)((*)((*)(-1, (exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (getindex)(x, 1)), (getindex)(x, 2)), (^)((exp)(ρ), 2))
            ˍ₋out[3] = (+)((*)((*)(-1, (/)((+)((*)(-1.0, (getindex)(x, 1)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ))), (getindex)(x, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)), (*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (^)((exp)(ρ), 2))), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (getindex)(x, 1)))
            ˍ₋out[4] = (/)((*)((*)((*)(-1, (exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (getindex)(x, 1)), (getindex)(x, 2)), (^)((exp)(ρ), 2))
            ˍ₋out[5] = (/)((*)((*)(-1, (^)((getindex)(x, 2), 2)), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (^)((exp)(ρ), 2))
            ˍ₋out[6] = (+)((*)((*)(-1, (/)((+)((*)(-1.0, (getindex)(x, 2)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ))), (getindex)(x, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)), (*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (^)((exp)(ρ), 2))), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (getindex)(x, 2)))
            ˍ₋out[7] = (+)((*)((*)(-1, (/)((+)((*)(-1.0, (getindex)(x, 1)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ))), (getindex)(x, 1))), (^)((exp)(ρ), 2))), (exp)(ρ)), (*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (^)((exp)(ρ), 2))), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (getindex)(x, 1)))
            ˍ₋out[8] = (+)((*)((*)(-1, (/)((+)((*)(-1.0, (getindex)(x, 2)), (*)((exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ))), (getindex)(x, 2))), (^)((exp)(ρ), 2))), (exp)(ρ)), (*)((*)((*)(-1, (/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (^)((exp)(ρ), 2))), (exp)((/)((+)((+)((*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1)), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (getindex)(x, 2)))
            ˍ₋out[9] = (+)((/)((+)((+)((+)((+)((+)((+)((*)(-2.0, (exp)((*)(2, ρ))), (*)((*)(-1.0, (log)(t)), (exp)(ρ))), (*)((*)((+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (^)((exp)(ρ), 2))), (exp)(ρ))), (log)(t)), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ))))), (*)((*)((exp)(ρ), (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)((exp)(ρ), (getindex)(x, 2)), (getindex)(θ, 2))), (*)((*)((*)((*)(-1.0, (+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (^)((exp)(ρ), 2))), (exp)(ρ)))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)((*)((*)(-1.0, (+)((/)((+)((*)(ρ, (exp)(ρ)), (exp)(ρ)), (exp)(ρ)), (*)((*)(-1, (/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (^)((exp)(ρ), 2))), (exp)(ρ)))), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (getindex)(x, 2)), (getindex)(θ, 2))), (exp)((*)(2, ρ))), (*)((*)(-2, (/)((+)((+)((+)((+)((+)((+)((*)(-1.0, (exp)((*)(2, ρ))), (*)((log)(t), (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ))))), (*)((*)(-1.0, (log)(t)), (exp)(ρ))), (*)((*)((exp)(ρ), (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)((exp)(ρ), (getindex)(x, 2)), (getindex)(θ, 2))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)((*)(-1.0, (exp)((/)((+)((+)((+)((*)(ρ, (exp)(ρ)), (*)((*)(-1, (getindex)(x, 1)), (getindex)(θ, 1))), (*)((*)(-1, (getindex)(x, 2)), (getindex)(θ, 2))), (log)(t)), (exp)(ρ)))), (getindex)(x, 2)), (getindex)(θ, 2))), (^)((exp)((*)(2, ρ)), 2))), (exp)((*)(2, ρ))))
        end
    end
    ˍ₋out
end


#=
                                  (exp((log(t) - x[1]*α[1] - x[2]*α[2]) / ρ)*x[1]) / ρ
                                  (exp((log(t) - x[1]*α[1] - x[2]*α[2]) / ρ)*x[2]) / ρ
 ((log(t) - x[1]*α[1] - x[2]*α[2])*exp((log(t) - x[1]*α[1] - x[2]*α[2]) / ρ)) / (ρ^2)
 =#






# df/dθ = df/dα dα/dθ (chain rule)

@variables α ρ t;
s = lsurv_weibull(α, ρ, t)
grad_symb = Symbolics.gradient(s, [α, ρ], simplify=true)
grad_symb = Symbolics.hessian(s, [α, ρ])

@variables α ρ t;
f = lpdf_weibull(α, ρ, t)
grad_symb = Symbolics.gradient(f, [α, ρ], simplify=true)
grad_symb = Symbolics.hessian(f, [α, ρ])



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








