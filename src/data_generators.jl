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
    dt1 = Weibull.(fill(0.75, n), inv.(exp.(-x .- z)))
    dt2 = Weibull.(fill(0.75, n), inv.(exp.(-x .- z)))
    t01 = rand.(rng, dt1)
    t02 = rand.(rng, dt2)
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