# to implement
# ID level robust variance estimators


using LSurvival, Random, Optim, BenchmarkTools, RCall

######################################################################
# residuals from fitted Cox models
######################################################################

#
"""
Remove the last element from an LSurvivalResp object
```julia
id, int, outt, data =
LSurvival.dgm(MersenneTwister(112), 100, 10; afun = LSurvival.int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))
wt ./= (sum(wt) / length(wt))

R = LSurvivalResp(int, outt, d, ID.(id))    # specification with ID only
Ri, Rj, idxi, idxj = pop(R);
```
"""
function pop(R::T) where {T<:LSurvivalResp}
    uid = unique(R.id)[end]
    idx = findall(getfield.(R.id, :value) .== uid.value)
    nidx = setdiff(collect(eachindex(R.id)),idx)
    Ri = LSurvivalResp(R.enter[idx], R.exit[idx], R.y[idx], R.wts[idx], R.id[idx]; origintime= R.origin)
    Rj = LSurvivalResp(R.enter[nidx], R.exit[nidx], R.y[nidx], R.wts[nidx], R.id[nidx],origintime= R.origin)
    Ri, Rj, idx, nidx
end

"""
id, int, outt, data =
LSurvival.dgm(MersenneTwister(112), 100, 10; afun = LSurvival.int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))
wt ./= (sum(wt) / length(wt))

P = PHParms(X)
R = LSurvivalResp(int, outt, d, ID.(id))    # specification with ID only
Ri, Rj, idxi, idxj = pop(R);
import Base.popat!
Pi = popat!(P, idxi, idxj)
"""
function popat!(P::T, idxi, idxj) where {T<:PHParms}
    Pi = PHParms(P.X[idxi,:], P._B, P._r[idxi], P._LL, P._grad, P._hess, 1, P.p)
    P.X =  P.X[idxj,:]
    P._r = P._r[idxj]
    P.n -= 1
    Pi
end

"""
Insert an observation into the front of an LSurvivalResp object
"""
function push!(Pi::T, Pj::T) where {T<:PHParms}
    Pj.X =  vcat(Pi.X, Pj.X)
    Pj._r = vcat(Pi._r, Pj._r)
    Pj.n +=1
    nothing
end


"""
Insert an observation into the front of an LSurvivalResp object

```julia
id, int, outt, data =
LSurvival.dgm(MersenneTwister(112), 100, 10; afun = LSurvival.int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))
wt ./= (sum(wt) / length(wt))

R = LSurvivalResp(int, outt, d, ID.(id))    # specification with ID only
Ri, Rj, idxi, idxj = pop(R);
R = push(Ri, Rj)
```

"""
function push(Ri::T, Rj::T) where {T<:LSurvivalResp}
    Ri = LSurvivalResp(
        vcat(Ri.enter, Rj.enter), 
        vcat(Ri.exit, Rj.exit), 
        vcat(Ri.y, Rj.y), 
        vcat(Ri.wts, Rj.wts), 
        vcat(Ri.id, Rj.id); 
        origintime= min(Ri.origin, Rj.origin))
end

function push(Pi::T, Pj::T) where {T<:PHParms}
end


"""
id, int, outt, data =
LSurvival.dgm(MersenneTwister(112), 100, 10; afun = LSurvival.int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))
wt ./= (sum(wt) / length(wt))
m = coxph(X,int, outt,d, wts=wt)

R::Union{Nothing,G}        # Survival response
P::L        # parameters
formula::Union{FormulaTerm,Nothing}
ties::String
fit::Bool
bh::Matrix{Float64}
RL::Union{Nothing,Vector{Matrix{Float64}}}        # residual matrix    



"""
function jackknife(m::M) where {M<:PHmodel}
    uid = unique(m.R.id)
    for i in eachindex(uid)
        Ri, Rj, idxi, idxj = pop(m.R);
        Pi = popat!(m.P, idxi, idxj)
        mi= PHModel(Rj, m.P, m.formula, m.ties, false, m.bh[1:Rj.eventtimes,:], nothing)
        fit!()
        m.R = push(Ri, Rj)
        push!(Pi, m.P)
    end
end
=#



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

