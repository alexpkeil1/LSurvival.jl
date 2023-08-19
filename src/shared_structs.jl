##################################################################################################################### 
# structs
#####################################################################################################################

abstract type AbstractLSurvID end
#abstract type AbstractLSurvIDtest <: Real end


DOC_ID = """
Type for identifying individuals in survival outcomes.

Accepts any Number or String

[ID(i) for i in 1:10]

Used for the id argument in 
    - Outcome types: LSurvResp, LSurvCompResp 
    - Model types: PHModel, KMRisk, AJRisk

"""
"""
$DOC_ID
"""
struct ID <: AbstractLSurvID
    id::T where {T<:Union{Number,String}}
end

function Base.show(io::IO, x::I) where {I<:ID}
    show(io, x.id)
end
Base.show(x::I) where {I<:ID} = Base.show(stdout, x::I)


"""
$DOC_LSURVRESP
"""
struct LSurvResp{
    E<:AbstractVector,
    X<:AbstractVector,
    Y<:AbstractVector,
    W<:AbstractVector,
    T<:Real,
    I<:AbstractLSurvID,
} <: AbstractLSurvResp
    enter::E
    "`exit`: Time at observation end"
    exit::X
    "`y`: event occurrence in observation"
    y::Y
    "`wts`: observation weights"
    wts::W
    "`eventtimes`: unique event times"
    eventtimes::E
    "`origin`: origin on the time scale"
    origin::T
    "`id`: person level identifier (must be wrapped in ID() function)"
    id::Vector{I}
end

function LSurvResp(
    enter::E,
    exit::X,
    y::Y,
    wts::W,
    id::Vector{I},
) where {
    E<:AbstractVector,
    X<:AbstractVector,
    Y<:AbstractVector,
    W<:AbstractVector,
    I<:AbstractLSurvID,
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
    origin = minimum(enter)
    if lw == 0
        wts = ones(Int64, ny)
    end

    return LSurvResp(enter, exit, y, wts, eventtimes, origin, id)
end

function LSurvResp(
    enter::E,
    exit::X,
    y::Y,
    id::Vector{I},
) where {E<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,I<:AbstractLSurvID}
    wts = similar(exit, 0)
    return LSurvResp(enter, exit, y, wts, id)
end

function LSurvResp(
    enter::E,
    exit::X,
    y::Y,
    wts::W,
) where {E<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,W<:AbstractVector}
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
        wts = ones(Int64, ny)
    end
    id = [ID(i) for i in eachindex(y)]
    return LSurvResp(enter, exit, y, wts, id)
end


function LSurvResp(
    enter::E,
    exit::X,
    y::Y,
) where {E<:AbstractVector,X<:AbstractVector,Y<:AbstractVector}
    wts = similar(exit, 0)
    return LSurvResp(enter, exit, y, wts)
end

function LSurvResp(exit::X, y::Y) where {X<:AbstractVector,Y<:AbstractVector}
    enter = zeros(eltype(exit), length(exit))
    return LSurvResp(enter, exit, y)
end

"""
$DOC_LSURVCOMPRESP
"""
struct LSurvCompResp{
    E<:AbstractVector,
    X<:AbstractVector,
    Y<:AbstractVector,
    W<:AbstractVector,
    T<:Real,
    I<:AbstractLSurvID,
    V<:AbstractVector,
    M<:AbstractMatrix,
} <: AbstractLSurvResp
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

function LSurvCompResp(
    enter::E,
    exit::X,
    y::Y,
    wts::W,
    id::Vector{I},
) where {
    E<:AbstractVector,
    X<:AbstractVector,
    Y<:AbstractVector,
    W<:AbstractVector,
    I<:AbstractLSurvID,
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
        throw(DimensionMismatch("wts must have length $n or length 0 but was $lw"))
    end
    eventtimes = sort(unique(exit[findall(y .> 0)]))
    origin = minimum(enter)
    if lw == 0
        wts = ones(Int64, ny)
    end
    eventtypes = sort(unique(y))
    eventmatrix = reduce(hcat, [y .== e for e in eventtypes[2:end]])

    return LSurvCompResp(
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

function LSurvCompResp(
    enter::E,
    exit::X,
    y::Y,
    id::Vector{I},
) where {E<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,I<:AbstractLSurvID}
    wts = ones(Int64, length(y))
    return LSurvCompResp(enter, exit, y, wts, id)
end

function LSurvCompResp(
    enter::E,
    exit::X,
    y::Y,
    wts::W,
) where {E<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,W<:AbstractVector}
    id = [ID(i) for i in eachindex(y)]
    return LSurvCompResp(enter, exit, y, wts, id)
end

function LSurvCompResp(
    enter::E,
    exit::X,
    y::Y,
) where {E<:AbstractVector,X<:AbstractVector,Y<:AbstractVector}
    id = [ID(i) for i in eachindex(y)]
    return LSurvCompResp(enter, exit, y, id)
end

function Base.show(io::IO, x<:AbstractLSurvResp; maxrows::Int = 10)
    lefttruncate = [e == x.origin ? "[" : "(" for e in x.enter]
    rightcensor = [y > 0 ? "]" : ")" for y in x.y]
    enter = [@sprintf("%.2g", e) for e in x.enter]
    exeunt = [@sprintf("%2.g", e) for e in x.exit]
    pr = [
        join([lefttruncate[i], enter[i], ",", exeunt[i], rightcensor[i]], "") for
        i in eachindex(exeunt)
    ]
    println("$(sum(x.y .> 0)) events, $(length(x.eventtimes)) unique event times")
    println("Origin: $(x.origin)")
    println("Max time: $(maximum(x.exit))")
    iob = IOBuffer()
    op = reduce(vcat, pr)
    nr = size(op, 1)
    if nr < maxrows
        [println(iob, "$(x.id[oo]). $(op1[oo])") for oo in eachindex(op)]
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

Base.show(x<:AbstractLSurvResp; kwargs...) = Base.show(stdout, x; kwargs...)
