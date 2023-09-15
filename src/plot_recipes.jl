"""
using Plots, LSurvival
```julia
dat4 = (
    id = [1, 1, 2, 2, 2, 3, 4, 5, 5, 6],
    enter = [1, 2, 5, 4, 6, 7, 3, 6, 8, 0],
    exit = [2, 5, 6, 7, 8, 9, 6, 8, 14, 9],
    status = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    x = [0.1, 0.1, 1.5, 1.5, 1.5, 0, 0, 0, 0, 3],
    z = [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
    w = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
)
R = LSurvivalResp(dat4.enter, dat4.exit, dat4.status)
plot([[R.enter[i], R.exit[i]] for i in eachindex(R.enter)], [[i, i] for i in values(R.id)])
```
"""
@recipe function f(r::LSurvivalResp; maxids = 10)
    # global
    xlabel --> "Time" # --> sets default
    ylabel --> "ID"
    label --> ""       # := over-rides user choices
    linecolor --> :black
    markershape --> :circle
    markerstrokecolor --> :black
    grid --> false
    plids = unique(r.id)[1:min(end, maxids)]
    plidxl = [findall(values(r.id) .== v) for v in unique(values(plids))]
    plidx = reduce(vcat, plidxl)
    @series begin
        seriestype := :path
        markershape := :none
        [[r.enter[i], r.exit[i]] for i = 1:length(r.enter[plidx])],
        [[v, v] for v in values(r.id)[plidx]]
    end
    @series begin
        seriestype := :scatter
        markercolor := ifelse.(r.y[plidx] .> 0, :black, :white)
        #markercolor := :black
        markersize --> 5
        [r.exit[i] for i = 1:length(r.enter[plidx])], values(r.id)[plidx]
    end
    # get the seriescolor passed by the user
    #c = get(plotattributes, :seriescolor, :auto)
    # highlight big errors, otherwise use the user-defined color
    # return data
    #[[r.enter[i], r.exit[i]] for i in eachindex(R.enter)], [[i, i] for i in values(r.id)]
end


"""
Recipe for kaplan meier curve
```julia
    using Plots, LSurvival
dat4 = (
    id = [1, 1, 2, 2, 2, 3, 4, 5, 5, 6],
    enter = [1, 2, 5, 4, 6, 7, 3, 6, 8, 0],
    exit = [2, 5, 6, 7, 8, 9, 6, 8, 14, 9],
    status = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    x = [0.1, 0.1, 1.5, 1.5, 1.5, 0, 0, 0, 0, 3],
    z = [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
    w = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
)
R = LSurvivalResp(dat4.enter, dat4.exit, dat4.status)
    k = kaplan_meier(dat4.enter, dat4.exit, dat4.status)
    plot(k)
```
    
"""
@recipe function f(k::KMSurv)
    # global
    xlabel --> "Time" # --> sets default
    ylabel --> "Survival"
    label --> ""       # := over-rides user choices
    linecolor --> :black
    markershape --> :circle
    markerstrokecolor --> :black
    grid --> false
    maxT = maximum(k.R.exit)
    minT = k.R.origin
    xlim --> (minT, maxT)
    ylim --> (0, 1)
    @series begin
        seriestype := :step
        markershape := :none
        vcat(minT, k.times, maxT), vcat(1, k.surv, minimum(k.surv))
    end
end


"""
Recipe for aalen-johansen risk curve


```julia
    using Plots, LSurvival
    res = z, x, outt, d, event, weights = LSurvival.dgm_comprisk(MersenneTwister(123123), 100)
    int = zeros(length(d)) # no late entry
    
        c = fit(AJSurv, int, outt, event)
        #risk2 = aalen_johansen(int, outt, event)
        plot(c)
```

"""
@recipe function f(c::AJSurv)
    # convenience munging
    maxT = maximum(c.R.exit)
    minT = c.R.origin
    # plot settings
    xlabel --> "Time" # --> sets default
    ylabel --> "Risk"
    label --> ""       # := over-rides user choices
    grid --> false
    xlim --> (minT, maxT)
    ylim --> (0, maximum(c.risk))
    for j in eachindex(c.R.eventtypes)[2:end]
        @series begin
            seriestype := :step
            markershape := :none
            label := c.R.eventtypes[j]
            vcat(minT, c.times, maxT), vcat(0, c.risk[:, j-1], maximum(c.risk[:, j-1]))
        end
    end
end


"""
Recipe for cox-model based risk curves

```julia
    using Plots, LSurvival
    res = z, x, outt, d, event, weights = LSurvival.dgm_comprisk(MersenneTwister(123123), 100)
    X = hcat(z, x)
    int = zeros(length(d)) # no late entry
    ft1 = fit(PHModel, X, int, outt, d .* (event .== 1))
    ft2 = fit(PHModel, X, int, outt, d .* (event .== 2))
    c = risk_from_coxphmodels([ft1, ft2], pred_profile = mean(X, dims=1))
    
    plot(c)
```

"""
@recipe function f(c::PHSurv)
    # convenience munging
    maxT = maximum(vcat(c.fitlist[1].R.exit, c.fitlist[2].R.exit))
    minT = min(c.fitlist[1].R.origin, c.fitlist[2].R.origin)
    # plot settings
    xlabel --> "Time" # --> sets default
    ylabel --> "Risk"
    label --> ""       # := over-rides user choices
    grid --> false
    xlim --> (minT, maxT)
    ylim --> (0, maximum(c.risk))
    for j in eachindex(c.eventtypes)
        @series begin
            seriestype := :step
            markershape := :none
            label := c.eventtypes[j]
            vcat(minT, c.times, maxT), vcat(0, c.risk[:, j], maximum(c.risk[:, j]))
        end
    end
end


# baseline hazard plot for Cox model
@userplot BaseHazPlot
"""
Plotting baseline hazard for a Cox model

```julia
using Plots, LSurvival
dat2 = (
    enter = [1, 2, 5, 2, 1, 7, 3, 4, 8, 8],
    exit = [2, 3, 6, 7, 8, 9, 9, 9, 14, 17],
    status = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    x = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
)
fte = coxph(@formula(Surv(enter, exit, status)~x), dat2, maxiter=0)
ftb = coxph(@formula(Surv(enter, exit, status)~x), dat2, ties="breslow", maxiter=0)

plot(fte, label="Efron")
plot!(ftb, label="Breslow")
```

"""
@recipe function f(h::BaseHazPlot)
    ft = h.args[1]
    # global
    xlabel --> "Time" # --> sets default
    ylabel --> "Cumulative hazard"
    label --> ""       # := over-rides user choices
    #linecolor --> :black
    grid --> false
    maxT = maximum(ft.R.exit)
    minT = ft.R.origin
    xlim --> (minT, maxT)
    #ylim --> (0, 1)
    cumbasehaz = cumsum(ft.bh[:, 1])
    times = ft.bh[:, 4]
    @series begin
        seriestype := :step
        markershape := :none
        vcat(minT, times, maxT), vcat(0, cumbasehaz, maximum(cumbasehaz))
    end
end



@userplot CoxDX
"""
```julia
using Plots, LSurvival
dat2 = (
    enter = [1, 2, 5, 2, 1, 7, 3, 4, 8, 8],
    exit = [2, 3, 6, 7, 8, 9, 9, 9, 14, 17],
    status = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    x = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
)
fte = coxph(@formula(Surv(enter, exit, status)~x), dat2)

coxdx(fte)

```
"""
@recipe function f(h::CoxDX; par=1)
    ft = h.args[1]
    time = ft.bh[:,1]
    nms = coefnames(ft)
    xlab --> "Time"
    ylab --> "Schoenfeld residual"
    res = residuals(ft, type = "schoenfeld")
    @series begin
        seriestype := :scatter
        shape := :circle
        markerstrokewidth := 0
        color --> :black
        label --> nms[par]
        markesize --> 2
        markeralpha --> 0.5
        time, res[:,par]
    end
end

@userplot CoxInfluence
"""
```julia
using Plots, LSurvival
dat2 = (
    enter = [1, 2, 5, 2, 1, 7, 3, 4, 8, 8],
    exit = [2, 3, 6, 7, 8, 9, 9, 9, 14, 17],
    status = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    x = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
)
fte = coxph(@formula(Surv(enter, exit, status)~x), dat2)

coxinfluence(fte, type="jackknife")

```
"""
@recipe function f(h::CoxInfluence; type="dfbeta")
    !issubset([type], ["dfbeta", "dfbetas", "jackknife"]) && throw("type must be 'dfbeta', 'dfbetas' or 'jackknife'")
    ft = h.args[1]
    id = values(ft.R.id)
    nms = coefnames(ft)
    xlab --> "ID"
    ylab --> "Residual"
    res = residuals(ft, type = type)
    grid --> false
    @series begin
        seriestype := :scatter
        shape := :circle
        markerstrokewidth := 0
        color --> :black
        label --> type
        markesize --> 2
        markeralpha --> 0.5
        id, res
    end
    @series begin
        seriestype := :hline
        label := ""
        color := :black
        [0]
    end
end

