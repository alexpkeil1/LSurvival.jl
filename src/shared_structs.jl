##################################################################################################################### 
# structs
#####################################################################################################################

#abstract type AbstractLSurvID <: Vector end
#abstract type AbstractLSurvIDtest <: Real end


abstract type SurvID <: Any end
  




"""
$DOC_LSURVRESP
"""
struct LSurvResp{
    E<:AbstractVector,
    X<:AbstractVector,
    Y<:AbstractVector,
    W<:AbstractVector,
    T<:Real,
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
end

function LSurvResp(
    enter::E,
    exit::X,
    y::Y,
    wts::Wf
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
    eventtimes = sort(unique(exit[findall(y .> 0)]))
    origin = minimum(enter)
    if lw == 0
        wts = ones(Int64, ny)
    end

    return LSurvResp(enter, exit, y, wts, eventtimes, origin)
end

function LSurvResp(
    enter::E,
    exit::X,
    y::Y,
) where {E<:AbstractVector,X<:AbstractVector,Y<:AbstractVector}
    wts = similar(exit, 0)
    return LSurvResp(enter, exit, y, wts)
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


function Base.show(io::IO, x::LSurvResp; maxrows::Int = 10)
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
        println(iob, op)
    else
        len = floor(Int, maxrows / 2)
        op1, op2 = deepcopy(op), deepcopy(op)
        op1 = op1[1:len]
        op2 = op2[(end-len+1):end]
        [println(iob, oo) for oo in op1]
        println(iob, "...")
        [println(iob, oo) for oo in op2]
    end
    str = String(take!(iob))
    println(io, str)
end

Base.show(x::LSurvResp; kwargs...) = Base.show(stdout, x; kwargs...)

