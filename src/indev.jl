# to implement
# - Greenwoods formula estimator for km 
# - robust standard error estimate for Cox model
# - using formulas

using LSurvival, Random


"""
```
id, int, outt, data =
LSurvival.dgm(MersenneTwister(1212), 20, 5; afun = LSurvival.int_0)

d, X = data[:, 4], data[:, 1:3]
weights = rand(length(d))

# survival outcome:
R = LSurvResp(int, outt, d, ID.(id))    # specification with ID only
```
"""
function bootstrap(rng::MersenneTwister, R::LSurvResp)
    uid = unique(R.id)
    bootid = sort(rand(rng, uid, length(uid)))
    idxl = [findall(getfield.(R.id, :value) .== bootidi.value) for bootidi in bootid]
    idx = reduce(vcat, idxl)
    nid = ID.(reduce(vcat, [fill(i, length(idxl[i])) for i in eachindex(idxl)]))
    R.id[idx]
    R2 = LSurvResp(R.enter[idx], R.exit[idx], R.y[idx], R.wts[idx], nid)
    idx, R2
end
bootstrap(R::LSurvResp) = bootstrap(MersenneTwister(), R::LSurvResp)


"""
```
z,x,t,d,event,weights =
LSurvival.dgm_comprisk(MersenneTwister(1212), 300)
enter = zeros(length(event))

# survival outcome:
R = LSurvCompResp(enter, t, event, weights, ID.(collect(1:length(t))))    # specification with ID only
```
"""
function bootstrap(rng::MersenneTwister, R::LSurvCompResp)
    uid = unique(R.id)
    bootid = sort(rand(rng, uid, length(uid)))
    idxl = [findall(getfield.(R.id, :value) .== bootidi.value) for bootidi in bootid]
    idx = reduce(vcat, idxl)
    nid = ID.(reduce(vcat, [fill(i, length(idxl[i])) for i in eachindex(idxl)]))
    R.id[idx]
    R2 = LSurvCompResp(R.enter[idx], R.exit[idx], R.y[idx], R.wts[idx], nid)
    idx, R2
end
bootstrap(R::LSurvCompResp) = bootstrap(MersenneTwister(), R::LSurvCompResp)

"""
```
using LSurvival, Random

id, int, outt, data =
LSurvival.dgm(MersenneTwister(1212), 20, 5; afun = LSurvival.int_0)

d, X = data[:, 4], data[:, 1:3]
weights = rand(length(d))

# survival outcome:
R = LSurvResp(int, outt, d, ID.(id))    # specification with ID only
P = PHParms(X)
idx, R2 = bootstrap(R)
P2 = bootstrap(idx, P)

Mod = PHModel(R2, P2)
LSurvival._fit!(Mod, start=Mod.P._B)

```

"""
function bootstrap(idx::Vector{Int}, P::PHParms)
    P2 = PHParms(P.X[idx, :])
    P2
end

"""
```
using LSurvival, Random

id, int, outt, data =
LSurvival.dgm(MersenneTwister(1212), 500, 5; afun = LSurvival.int_0)

d, X = data[:, 4], data[:, 1:3]
weights = rand(length(d))

# survival outcome:
R = LSurvResp(int, outt, d, ID.(id))    # specification with ID only
P = PHParms(X)

Mod = PHModel(R, P)
LSurvival._fit!(Mod, start=Mod.P._B)


# careful propogation of bootstrap sampling
idx, R2 = bootstrap(R)
P2 = bootstrap(idx, P)
Modb = PHModel(R2, P2)
LSurvival._fit!(Mod, start=Mod.P._B)

# convenience function for bootstrapping a model
Modc = bootstrap(Mod)
LSurvival._fit!(Modc, start=Modc.P._B)
Modc.P.X = nothing
Modc.R = nothing

```
"""
function bootstrap(rng::MersenneTwister, m::PHModel)
    idx, R2 = bootstrap(rng, m.R)
    P2 = bootstrap(idx, m.P)
    PHModel(R2, P2, m.ties, false, m.bh)
end
bootstrap(m::PHModel) = bootstrap(MersenneTwister(), m::PHModel)




# in progress functions
# taken from GLM.jl/src/linpred.jl
function modelframe(f::FormulaTerm, data, contrasts::AbstractDict, ::Type{M}) where {M}
    Tables.istable(data) ||
        throw(ArgumentError("expected data in a Table, got $(typeof(data))"))
    t = Tables.columntable(data)
    msg = StatsModels.checknamesexist(f, t)
    msg != "" && throw(ArgumentError(msg))
    data, _ = StatsModels.missing_omit(t, f)
    sch = schema(f, data, contrasts)
    f = apply_schema(f, sch, M)
    f, modelcols(f, data)
end


function fit(
    ::Type{M},
    f::FormulaTerm,
    data;
    ties = "breslow",
    id::AbstractVector{<:AbstractLSurvID} = [ID(i) for i in eachindex(y)],
    wts::AbstractVector{<:Real} = similar(y, 0),
    offset::AbstractVector{<:Real} = similar(y, 0),
    contrasts::AbstractDict{Symbol} = Dict{Symbol,Any}(),
    fitargs...,
) where {M<:AbstractPH}

    f, (y, X) = modelframe(f, data, contrasts, M)

    # Check that X and y have the same number of observations
    #if size(X, 1) != size(y, 1)
    #    throw(DimensionMismatch("number of rows in X and y must match"))
    #end

    #rr = GlmResp(y, d, l, off, wts)

    #res = M(rr, X, nothing, false)
    R = LSurvResp(enter, exit, y, wts)
    P = PHParms(X)

    res = M(R, P, ties)

    #return coxmodel(_in::Array{<:Real,1}, 
    #          _out::Array{<:Real,1}, 
    #          d::Array{<:Real,1}, 
    #          X::Array{<:Real,2}; weights=nothing, method="efron", inits=nothing , tol=10e-9,maxiter=500)
    return fit!(res; fitargs...)
end