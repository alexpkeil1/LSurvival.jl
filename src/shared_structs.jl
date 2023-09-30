##################################################################################################################### 
# structs
#####################################################################################################################

abstract type AbstractLSurvivalID end
abstract type AbstractSurvTime end

"""
$DOC_ID
"""
struct ID <: AbstractLSurvivalID
    value::T where {T<:Union{Number,String}}
end

"""
$DOC_STRATA
"""
struct Strata <: AbstractLSurvivalID
    value::T where {T<:Union{Number,String}}
end

function Base.values(x::Vector{I}) where {I<:AbstractLSurvivalID}
    [xi.value for xi in x]
end

function Base.show(io::IO, x::I) where {I<:AbstractLSurvivalID}
    show(io, x.value)
end
Base.show(x::I) where {I<:AbstractLSurvivalID} = Base.show(stdout, x::I)

function Base.isless(x::I, y::I) where {I<:AbstractLSurvivalID}
    Base.isless(x.value, y.value)
end

function Base.length(x::I) where {I<:AbstractLSurvivalID}
    Base.length(x.value)
end


struct Surv{E<:Real,X<:Real,Y<:Real,O<:Real} <: AbstractSurvTime
    enter::E
    exit::X
    y::Y
    origin::O
end

function Surv(enter::E, exit::X, y::Y, origintime = nothing) where {E<:Real,X<:Real,Y<:Real}
    origin = isnothing(origintime) ? zero(E) : origintime
    return Surv(enter, exit, y, origin)
end

function Surv(exit::X, y::Y; kwargs...) where {X<:Real,Y<:Real}
    return Surv(zero(X), exit, y; kwargs...)
end

function Surv(exit::X; kwargs...) where {X<:Real}
    return Surv(zero(X), exit, 1; kwargs...)
end



"""
$DOC_LSURVRESP
"""
struct LSurvivalResp{
    E<:Vector,
    X<:Vector,
    Y<:Union{Vector{<:Real},BitVector},
    W<:Vector,
    T<:Real,
    I<:AbstractLSurvivalID,
} <: AbstractLSurvivalResp
    enter::E
    "`exit`: Time at observation end"
    exit::X
    "`y`: event occurrence in observation"
    y::Y
    "`wts`: observation weights"
    wts::W
    "`eventtimes`: unique event times"
    eventtimes::X
    "`origin`: origin on the time scale"
    origin::T
    "`id`: person level identifier (must be wrapped in ID() function)"
    id::Vector{I}
end

function LSurvivalResp(
    enter::E,
    exit::X,
    y::Y,
    wts::W,
    id::Vector{I};
    origintime = nothing,
) where {
    E<:Vector,
    X<:Vector,
    Y<:Union{Vector{<:Real},BitVector},
    W<:Vector,
    I<:AbstractLSurvivalID,
}
    ne = length(enter)
    nx = length(exit)
    ny = length(y)
    lw = length(wts)
    if !(ne == nx == ny)
        throw(
            DimensionMismatch(
                "lengths of enter, exit, and y ($ne, $nx, $ny) are not equal",
            ),
        )
    end
    if lw != 0 && lw != ny
        throw(DimensionMismatch("wts must have length $ny or length 0 but was $lw"))
    end
    eventtimes = sort(unique(exit[findall(y .> 0)]))
    origin = isnothing(origintime) ? minimum(enter) : origintime
    if lw == 0
        wts = ones(Int, ny)
    end

    return LSurvivalResp(enter, exit, y, wts, eventtimes, origin, id)
end

function LSurvivalResp(
    y::Vector{Y},
    wts::W,
    id::Vector{I};
    kwargs...,
) where {Y<:AbstractSurvTime,W<:Vector,I<:AbstractLSurvivalID}
    enter = [yi.enter for yi in y]
    exit = [yi.exit for yi in y]
    d = [yi.y for yi in y]
    return LSurvivalResp(enter, exit, d, wts, id; kwargs...)
end


function LSurvivalResp(
    enter::E,
    exit::X,
    y::Y,
    id::Vector{I};
    kwargs...,
) where {E<:Vector,X<:Vector,Y<:Union{Vector{<:Real},BitVector},I<:AbstractLSurvivalID}
    wts = similar(exit, 0)
    return LSurvivalResp(enter, exit, y, wts, id; kwargs...)
end

function LSurvivalResp(
    enter::E,
    exit::X,
    y::Y,
    wts::W;
    kwargs...,
) where {E<:Vector,X<:Vector,Y<:Union{Vector{<:Real},BitVector},W<:Vector}
    ne = length(enter)
    nx = length(exit)
    ny = length(y)
    lw = length(wts)
    if !(ne == nx == ny)
        throw(
            DimensionMismatch(
                "lengths of enter, exit, and y ($ne, $nx, $ny) are not equal",
            ),
        )
    end
    if lw != 0 && lw != ny
        throw(DimensionMismatch("wts must have length $ny or length 0 but was $lw"))
    end
    if lw == 0
        wts = ones(Int, ny)
    end
    id = [ID(i) for i in eachindex(y)]
    return LSurvivalResp(enter, exit, y, wts, id; kwargs...)
end


function LSurvivalResp(
    enter::E,
    exit::X,
    y::Y;
    kwargs...,
) where {E<:Vector,X<:Vector,Y<:Union{Vector{<:Real},BitVector}}
    wts = similar(exit, 0)
    return LSurvivalResp(enter, exit, y, wts; kwargs...)
end

function LSurvivalResp(exit::X, y::Y; kwargs...) where {X<:Vector,Y<:Vector}
    enter = zeros(eltype(exit), length(exit))
    return LSurvivalResp(enter, exit, y; kwargs...)
end

"""
$DOC_LSURVCOMPRESP
"""
struct LSurvivalCompResp{
    E<:Vector,
    X<:Vector,
    Y<:Union{Vector{<:Real},BitVector},
    W<:Vector,
    I<:AbstractLSurvivalID,
    V<:Vector,
    M<:AbstractMatrix,
    T<:Real,
} <: AbstractLSurvivalResp
    enter::E
    "`exit`: Time at observation end"
    exit::X
    "`y`: event type in observation (integer)"
    y::Y
    "`wts`: observation weights"
    wts::W
    "`eventtimes`: unique event times"
    eventtimes::X
    "`origin`: origin on the time scale"
    origin::T
    "`id`: person level identifier (must be wrapped in ID() function)"
    id::Vector{I}
    "`eventtypes`: vector of unique event types"
    eventtypes::V
    "`eventmatrix`: matrix of indicators on the observation level"
    eventmatrix::M
end

function LSurvivalCompResp(
    enter::E,
    exit::X,
    y::Y,
    wts::W,
    id::Vector{I};
    origintime = nothing,
    etypes = ones(0),
    ematrix = ones(0,0),
) where {
    E<:Vector,
    X<:Vector,
    Y<:Union{Vector{<:Real},BitVector},
    W<:Vector,
    I<:AbstractLSurvivalID,
}
    ne = length(enter)
    nx = length(exit)
    ny = length(y)
    lw = length(wts)
    if !(ne == nx == ny)
        throw(
            DimensionMismatch(
                "lengths of enter, exit, and y ($ne, $nx, $ny) are not equal",
            ),
        )
    end
    if lw != 0 && lw != ny
        throw(DimensionMismatch("wts must have length $ny or length 0 but was $lw"))
    end
    eventtimes = sort(unique(exit[findall(y .> 0)]))

    origin = isnothing(origintime) ? minimum(enter) : origintime
    if lw == 0
        wts = ones(Int, ny)
    end
    eventtypes = length(etypes) == 0 ? sort(unique(y)) : etypes
    length(eventtypes) < length(sort(unique(y))) &&
        throw(DimensionMismatch("etypes cannot be have fewer unique values than y"))
    eventmatrix = size(ematrix,1) == 0 ? reduce(hcat, [y .== e for e in eventtypes[2:end]]) : ematrix

    return LSurvivalCompResp(
        enter,
        exit,
        y,
        wts,
        eventtimes,
        origin,
        id,
        eventtypes,
        eventmatrix,
    )
end

function LSurvivalCompResp(
    y::Vector{Y},
    wts::W,
    id::Vector{I};
    kwargs...,
) where {Y<:AbstractSurvTime,W<:Vector,I<:AbstractLSurvivalID}
    enter = [yi.enter for yi in y]
    exit = [yi.exit for yi in y]
    event = [yi.y for yi in y]
    return LSurvivalCompResp(enter, exit, event, wts, id; kwargs...)
end



function LSurvivalCompResp(
    enter::E,
    exit::X,
    y::Y,
    id::Vector{I};
    kwargs...,
) where {E<:Vector,X<:Vector,Y<:Union{Vector{<:Real},BitVector},I<:AbstractLSurvivalID}
    wts = ones(Int, length(y))
    return LSurvivalCompResp(enter, exit, y, wts, id; kwargs...)
end



function LSurvivalCompResp(
    enter::E,
    exit::X,
    y::Y,
    wts::W;
    kwargs...,
) where {E<:Vector,X<:Vector,Y<:Union{Vector{<:Real},BitVector},W<:Vector}
    id = [ID(i) for i in eachindex(y)]
    return LSurvivalCompResp(enter, exit, y, wts, id; kwargs...)
end

function LSurvivalCompResp(
    enter::E,
    exit::X,
    y::Y;
    kwargs...,
) where {E<:Vector,X<:Vector,Y<:Union{Vector{<:Real},BitVector}}
    id = [ID(i) for i in eachindex(y)]
    return LSurvivalCompResp(enter, exit, y, id; kwargs...)
end

function LSurvivalCompResp(
    exit::X,
    y::Y;
    kwargs...,
) where {X<:Vector,Y<:Union{Vector{<:Real},BitVector}}
    return LSurvivalCompResp(zeros(length(exit)), exit, y; kwargs...)
end


function Base.show(io::IO, x::T; maxrows::Int = 10) where {T<:AbstractLSurvivalResp}
    lefttruncate = [e == x.origin ? "[" : "(" for e in x.enter]
    rightcensor = [y > 0 ? "]" : ")" for y in x.y]
    enter = [@sprintf("%.2g", e) for e in x.enter]
    exeunt = [@sprintf("%.2g", e) for e in x.exit]
    pr = [
        join([lefttruncate[i], enter[i], ",", exeunt[i], rightcensor[i]], "") for
        i in eachindex(exeunt)
    ]
    println("$(sum(x.y .> 0)) events, $(length(x.eventtimes)) unique event times")
    println("Origin: $(x.origin)")
    println("Max time: $(maximum(x.exit))")
    iob = IOBuffer()
    op = reduce(vcat, pr)
    nr = typeof(op) == String ? 1 : size(op, 1)
    if nr == 1
        println(iob, "$(x.id[1]). $(op)")
    elseif nr < maxrows
        [println(iob, "$(x.id[oo]). $(op[oo])") for oo in eachindex(op)]
    else
        len = floor(Int, maxrows / 2)
        op1, op2 = deepcopy(op), deepcopy(op)
        op1 = op1[1:len]
        op2 = op2[(end-len+1):end]
        [println(iob, "$(x.id[1:len][oo]). $(op1[oo])") for oo in eachindex(op1)]
        println(iob, "...")
        [println(iob, "$(x.id[(end-len+1):end][oo]). $(op2[oo])") for oo in eachindex(op2)]
    end
    str = String(take!(iob))
    println(io, str)
end

Base.show(x::T; kwargs...) where {T<:AbstractLSurvivalResp} =
    Base.show(stdout, x; kwargs...)

function Base.show(io::IO, x::T) where {T<:AbstractSurvTime}
    lefttruncate = x.enter == x.origin ? "[" : "("
    rightcensor = x.y > x.origin ? "]" : ")"
    enter = @sprintf("%.2g", x.enter)
    exeunt = @sprintf("%.2g", x.exit)
    pr = join([lefttruncate, enter, ",", exeunt, rightcensor], "")
    print(io, pr)
end

Base.show(x::T; kwargs...) where {T<:AbstractSurvTime} = Base.show(stdout, x; kwargs...)



Base.length(x::LSurvivalCompResp) = length(x.exit)
Base.length(x::LSurvivalResp) = length(x.exit)

function survcheck(y::Y) where {Y<:AbstractLSurvivalResp}
    @assert all(y.exit .> y.enter) "Error detected in survival times: some entry times are equal to or greater than exit times; expecting: (entry<exit)"
end
