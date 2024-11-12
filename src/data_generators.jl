expit(mu) = inv(1.0 + exp(-mu))

function int_nc(v, l, a)
    expit(-1.0 + 3 * v + 2 * l)
end

function int_0(v, l, a)
    0.1
end

function lprob(v, l, a)
    expit(-3 + 2 * v + 0 * l + 0 * a)
end

function yprob(v, l, a)
    expit(-3 + 2 * v + 0 * l + 2 * a)
end


"""
$DOC_DGM
"""
function dgm(rng::MersenneTwister, n::Int, maxT::Int; afun = int_0, yfun = yprob, lfun = lprob)
    V = rand(rng, n)
    LAY = Array{Float64,2}(undef, n * maxT, 4)
    keep = ones(Bool, n * maxT)
    id = sort(reduce(vcat, fill(collect(1:n), maxT)))
    time = (reduce(vcat, fill(collect(1:maxT), n)))
    for i = 1:n
        v = V[i]
        l = 0
        a = 0
        lkeep = true
        for t = 1:maxT
            currIDX = (i - 1) * maxT + t
            l = lfun(v, l, a) > rand(rng) ? 1 : 0
            a = afun(v, l, a) > rand(rng) ? 1 : 0
            y = yfun(v, l, a) > rand(rng) ? 1 : 0
            LAY[currIDX, :] .= [v, l, a, y]
            keep[currIDX] = lkeep
            lkeep = (!lkeep || (y == 1)) ? false : true
        end
    end
    id[findall(keep)], time[findall(keep)] .- 1, time[findall(keep)], LAY[findall(keep), :]
end
dgm(n::Int, maxT::Int; kwargs...) = dgm(MersenneTwister(), n::Int, maxT::Int; kwargs...)


"""
$DOC_DGM_COMPRISK
"""
function dgm_comprisk(rng::MersenneTwister, n::Int)
    z = rand(rng, n) .* 5
    x = rand(rng, n) .* 5
    #dt1 = Weibull.(fill(0.75, n), inv.(exp.(-x .- z)))
    #dt2 = Weibull.(fill(0.75, n), inv.(exp.(-x .- z)))
    #t01 = rand.(rng, dt1)
    #t02 = rand.(rng, dt2)
    h1 = inv.(exp.(-x .- z))
    h2 = inv.(exp.(-x .- z))
    t01 = [randweibull(rng, .75, hi) for hi in h1]
    t02 = [randweibull(rng, .75, hi) for hi in h2]
    t0 = min.(t01, t02)
    t = Array{Float64,1}(undef, n)
    for i = 1:n
        t[i] = t0[i] > 1.0 ? 1.0 : t0[i]
    end
    d = (t .== t0)
    event = (t .== t01) .+ 2.0 .* (t .== t02)
    weightsu = rand(rng, n) .* 5.0
    weights = weightsu ./ mean(weightsu)
    reshape(round.(z, digits = 4), (n, 1)),
    reshape(round.(x, digits = 4), (n, 1)),
    round.(t, digits = 4),
    d,
    event,
    round.(weights, digits = 4)
end
dgm_comprisk(n) = dgm_comprisk(MersenneTwister(), n)

"""
Proportional hazards model from a Weibull distribution with
scale parameter λ

```
dgm_phmodel(rng::MersenneTwister, n::Int; 
    λ=1.25,
    β=[0.0, 0.0]
    )
```
keyword parameters:
  - λ: Weibull scale parameter
  - β: vector of regression coefficients

```
rng = MersenneTwister()
X, t, d, _ = dgm_phmodel(2000; λ=1.25,β=[1.0, -0.5])
coxph(@formula(Surv(t0,t,d)~x+z), (t=t,t0=t.*0,d=d,x=X[:,1],z=X[:,2]))
```
"""
function dgm_phmodel(rng::MersenneTwister, n::Int; 
    λ=1.25,
    β=[0.0, 0.0]
    )
    X = rand(n, length(β))
    r = (exp.(X * β)).^(-1/λ)
    t0 = [LSurvival.randweibull(rng, λ, ri) for ri in r]
    t = Array{Float64,1}(undef, n)
    for i = 1:n
        t[i] = t0[i] > 1.0 ? 1.0 : t0[i]
    end
    d = (t .== t0)
    event = (t .== t0)
    weightsu = rand(rng, n) .* 5.0
    weights = weightsu ./ mean(weightsu)
    #
    X,
    t,
    d,
    event,
    round.(weights, digits = 4)
end
dgm_phmodel(n;kwargs...) = dgm_phmodel(MersenneTwister(), n;kwargs...)



