# in progress functions

function fit(::Type{M},
                 f::FormulaTerm,
                 data;
                 wts::AbstractVector{<:Real}      = similar(y, 0),
                 offset::AbstractVector{<:Real}   = similar(y, 0),
                 method::Symbol = :cholesky,
                 dofit::Union{Bool, Nothing} = nothing,
                 contrasts::AbstractDict{Symbol}=Dict{Symbol,Any}(),
                 fitargs...) where {M<:AbstractPH}
    
        f, (y, X) = modelframe(f, data, contrasts, M)
    
        # Check that X and y have the same number of observations
        #if size(X, 1) != size(y, 1)
        #    throw(DimensionMismatch("number of rows in X and y must match"))
        #end
    
        #rr = GlmResp(y, d, l, off, wts)
        
        #res = M(rr, X, nothing, false)
        R = LSurvResp(enter, exit, y, wts)
        P = PHParms(X)
   
        res = M(R,P, ties)
  
        #return coxmodel(_in::Array{<:Real,1}, 
        #          _out::Array{<:Real,1}, 
        #          d::Array{<:Real,1}, 
        #          X::Array{<:Real,2}; weights=nothing, method="efron", inits=nothing , tol=10e-9,maxiter=500)
        return fit!(res; fitargs...)
    end