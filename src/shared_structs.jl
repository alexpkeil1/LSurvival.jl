#= #################################################################################################################### 
structs
=# ####################################################################################################################

  struct LSurvResp{E<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,W<:AbstractVector, T<:Real} <: AbstractLSurvResp 
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
    
  function LSurvResp(enter::E, exit::X, y::Y, wts::W) where {E<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,W<:AbstractVector}
    ne  = length(enter)
    nx = length(exit)
    ny = length(y)
    lw = length(wts)
    if !(ne == nx == ny)
        throw(DimensionMismatch("lengths of enter, exit, and y ($ne, $nx, $ny) are not equal"))
    end
    if lw != 0 && lw != ny
        throw(DimensionMismatch("wts must have length $n or length 0 but was $lw"))
    end
    eventtimes = sort(unique(exit[findall(y .> 0)]))
    origin = minimum(enter)
    if lw == 0
      wts = ones(Int64,ny)
    end
   
    return LSurvResp(enter,exit,y,wts,eventtimes,origin)
  end
  
  # import Base.show
  # using Printf
  # resp = LSurvResp(enter, t, d, ones(length(d)))
  function show(io::IO, x::LSurvResp)
    lefttruncate = [ e == x.origin ? "[" : "(" for e in x.enter]
    rightcensor = [ y > 0 ? "]" : ")" for y in x.y]
    enter = [@sprintf("%.2g", e) for e in x.enter]
    exit = [@sprintf("%2.g", e) for e in x.exit]
    pr = [join([lefttruncate[i], enter[i], ",", exit[i], rightcensor[i]], "") for i in 1:length(exit)]
    println("$(sum(x.y .> 0)) events, $(length(x.eventtimes)) unique event times")
    println("Origin: $(x.origin) events, Max time: $(maximum(x.exit))")
    show(io, reduce(vcat, pr))
  end
    
  show(x::LSurvResp) = show(stdout, x)

