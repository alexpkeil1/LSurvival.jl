# to implement
# - Greenwoods formula estimator for km 
# - robust standard error estimate for Cox model
# - using formulas

if false
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
function bootstrap(rng::MersenneTwister(), R::LSurvResp)
  uid = unique(R.id)
  bootid = sort(rand(uid, length(uid)))
  idx = [findall(R.id .== bootidi) for bootidi in bootid]


end


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