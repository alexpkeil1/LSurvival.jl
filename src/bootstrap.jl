"""
$DOC_BOOTSTRAP_SURVRESP
"""
function bootstrap(rng::MersenneTwister, R::T) where {T<:LSurvivalResp}
    uid = unique(R.id)
    bootid = sort(rand(rng, uid, length(uid)))
    idxl = [findall(getfield.(R.id, :value) .== bootidi.value) for bootidi in bootid]
    idx = reduce(vcat, idxl)
    nid = ID.(reduce(vcat, [fill(i, length(idxl[i])) for i in eachindex(idxl)]))
    R.id[idx]
    R2 = LSurvivalResp(R.enter[idx], R.exit[idx], R.y[idx], R.wts[idx], nid)
    idx, R2
end
bootstrap(R::T) where {T<:LSurvivalResp} = bootstrap(MersenneTwister(), R::T)



"""
$DOC_BOOTSTRAP_SURVCOMPRESP
"""
function bootstrap(rng::MersenneTwister, R::T) where {T<:LSurvivalCompResp}
    uid = unique(R.id)
    bootid = sort(rand(rng, uid, length(uid)))
    idxl = [findall(getfield.(R.id, :value) .== bootidi.value) for bootidi in bootid]
    idx = reduce(vcat, idxl)
    nid = ID.(reduce(vcat, [fill(i, length(idxl[i])) for i in eachindex(idxl)]))
    R.id[idx]
    R2 = LSurvivalCompResp(R.enter[idx], R.exit[idx], R.y[idx], R.wts[idx], nid)
    idx, R2
end
bootstrap(R::T) where {T<:LSurvivalCompResp} = bootstrap(MersenneTwister(), R::T)


"""
$DOC_BOOTSTRAP_PHPARMS
"""
function bootstrap(idx::Vector{Int}, P::S) where {S<:PSParms}
    P2 = PSParms(P.X[idx, :])
    P2
end
function bootstrap(idx::Vector{Int}, P::S) where {S<:PHParms}
    P2 = PHParms(P.X[idx, :])
    P2
end



"""
$DOC_BOOTSTRAP_PHMODEL
"""
function bootstrap(rng::MersenneTwister, m::M) where {M<:PHModel}
    idx, R2 = bootstrap(rng, m.R)
    P2 = bootstrap(idx, m.P)
    PHModel(R2, P2, m.formula, m.ties, false, m.bh)
end
bootstrap(m::PHModel) = bootstrap(MersenneTwister(), m)

function bootstrap(rng::MersenneTwister, m::M) where {M<:PSModel}
    idx, R2 = bootstrap(rng, m.R)
    P2 = bootstrap(idx, m.P)
    PSModel(R2, P2, m.formula, m.d, false)
end
bootstrap(m::M) where {M<:PSModel} = bootstrap(MersenneTwister(), m)



"""
$DOC_BOOTSTRAP_PHMODEL
"""
function bootstrap(rng::MersenneTwister, m::PHModel, iter::Int; kwargs...)
    if isnothing(m.R) || isnothing(m.P.X)
        throw(
            "Model is missing response or predictor matrix, use keepx=true, keepy=true for original fit",
        )
    end
    res = zeros(iter, length(coef(m)))
    @inbounds for i = 1:iter
        mb = bootstrap(rng, m)
        LSurvival._fit!(mb; kwargs...)
        @debug "Log-partial-likelihood $i: $(mb.P._LL[1])"
        res[i, :] = coef(mb)
    end
    res
end
bootstrap(m::PHModel, iter::Int; kwargs...) =
    bootstrap(MersenneTwister(), m, iter; kwargs...)


DOC_BOOTSTRAP_PSMODEL="""
Bootstrap methods for parametric survival models

Signatures

```julia
bootstrap(rng::MersenneTwister, m::M, iter::Int; kwargs...) where {M<:PSModel}
bootstrap(m::M, iter::Int; kwargs...) where {M<:PSModel}
bootstrap(rng::MersenneTwister, m::M) where {M<:PSModel}
bootstrap(m::M) where {M<:PSModel}
```
"""   
function bootstrap(rng::MersenneTwister, m::M, iter::Int; kwargs...) where {M<:PSModel}
    if isnothing(m.R) || isnothing(m.P.X)
        throw(
            "Model is missing response or predictor matrix, use keepx=true, keepy=true for original fit",
        )
    end
    res = zeros(iter, length(params(m)))
    @inbounds for i = 1:iter
        mb = bootstrap(rng, m)
        LSurvival._fit!(mb; kwargs...)
        @debug "Log-partial-likelihood $i: $(mb.P._LL[1])"
        res[i, :] = params(mb)
    end
    res
end

bootstrap(m::M, iter::Int; kwargs...) where {M<:PSModel} =
    bootstrap(MersenneTwister(), m, iter; kwargs...)




"""
$DOC_BOOTSTRAP_KMSURV
"""
function bootstrap(rng::MersenneTwister, m::M; kwargs...) where {M<:KMSurv}
    _, R2 = bootstrap(rng, m.R)
    boot = KMSurv(R2)
    LSurvival._fit!(boot; kwargs...)
end
bootstrap(m::M; kwargs...) where {M<:KMSurv} = bootstrap(MersenneTwister(), m; kwargs...)

function bootstrap(rng::MersenneTwister, m::M, iter::Int; kwargs...) where {M<:KMSurv}
    if isnothing(m.R)
        throw("Model is missing response matrix, use keepy=true for original fit")
    end
    res = zeros(iter, 1)
    @inbounds for i = 1:iter
        mb = bootstrap(rng, m)
        LSurvival._fit!(mb; kwargs...)
        res[i, :] = mb.surv[end:end]
    end
    res
end
bootstrap(m::M, iter::Int; kwargs...) where {M<:KMSurv} =
    bootstrap(MersenneTwister(), m, iter::Int; kwargs...)




"""
$DOC_BOOTSTRAP_AJSURV
"""
function bootstrap(rng::MersenneTwister, m::M; kwargs...) where {M<:AJSurv}
    _, R2 = bootstrap(rng, m.R)
    boot = AJSurv(R2)
    LSurvival._fit!(boot; kwargs...)
end
bootstrap(m::M; kwargs...) where {M<:AJSurv} = bootstrap(MersenneTwister(), m; kwargs...)

function bootstrap(rng::MersenneTwister, m::M, iter::Int; kwargs...) where {M<:AJSurv}
    if isnothing(m.R)
        throw("Model is missing response matrix, use keepy=true for original fit")
    end
    res = zeros(iter, size(m.risk, 2))
    @inbounds for i = 1:iter
        mb = bootstrap(rng, m)
        LSurvival._fit!(mb; kwargs...)
        res[i, :] = mb.risk[end, :]
    end
    res
end
bootstrap(m::M, iter::Int; kwargs...) where {M<:AJSurv} =
    bootstrap(MersenneTwister(), m, iter::Int; kwargs...)

