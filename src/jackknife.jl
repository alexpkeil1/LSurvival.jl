"""
Remove the last element from an LSurvivalResp object
```julia
id, int, outt, data =
LSurvival.dgm(MersenneTwister(112), 100, 10; afun = LSurvival.int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))
wt ./= (sum(wt) / length(wt))

R = LSurvivalResp(int, outt, d, ID.(id))    # specification with ID only
Ri, Rj, idxi, idxj = pop(R);
```
"""
function pop(R::T) where {T<:LSurvivalResp}
    uid = unique(R.id)[end]
    idx = findall(getfield.(R.id, :value) .== uid.value)
    nidx = setdiff(collect(eachindex(R.id)),idx)
    Ri = LSurvivalResp(R.enter[idx], R.exit[idx], R.y[idx], R.wts[idx], R.id[idx]; origintime= R.origin)
    Rj = LSurvivalResp(R.enter[nidx], R.exit[nidx], R.y[nidx], R.wts[nidx], R.id[nidx],origintime= R.origin)
    Ri, Rj, idx, nidx
end

"""
id, int, outt, data =
LSurvival.dgm(MersenneTwister(112), 100, 10; afun = LSurvival.int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))
wt ./= (sum(wt) / length(wt))

P = PHParms(X)
R = LSurvivalResp(int, outt, d, ID.(id))    # specification with ID only
Ri, Rj, idxi, idxj = pop(R);
Pi = popat!(P, idxi, idxj)
"""
function popat!(P::T, idxi, idxj) where {T<:PHParms}
    Pi = PHParms(P.X[idxi,:], P._B, P._r[idxi], P._LL, P._grad, P._hess, 1, P.p)
    P.X =  P.X[idxj,:]
    P._r = P._r[idxj]
    P.n -= length(idxi)
    Pi
end



"""
Insert an observation into the front of an LSurvivalResp object

```julia
id, int, outt, data =
LSurvival.dgm(MersenneTwister(112), 100, 10; afun = LSurvival.int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))
wt ./= (sum(wt) / length(wt))

R = LSurvivalResp(int, outt, d, ID.(id))    # specification with ID only
Ri, Rj, idxi, idxj = pop(R);
R = push(Ri, Rj)
```

"""
function push(Ri::T, Rj::T) where {T<:LSurvivalResp}
    Ri = LSurvivalResp(
        vcat(Ri.enter, Rj.enter), 
        vcat(Ri.exit, Rj.exit), 
        vcat(Ri.y, Rj.y), 
        vcat(Ri.wts, Rj.wts), 
        vcat(Ri.id, Rj.id); 
        origintime= min(Ri.origin, Rj.origin))
end


"""
Insert an observation into the front of an PHParms object
"""
function push!(Pi::T, Pj::T) where {T<:PHParms}
    Pj.X =  vcat(Pi.X, Pj.X)
    Pj._r = vcat(Pi._r, Pj._r)
    Pj.n +=1
    nothing
end


"""
id, int, outt, data =
LSurvival.dgm(MersenneTwister(112), 100, 10; afun = LSurvival.int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))
wt ./= (sum(wt) / length(wt))
m = coxph(X,int, outt,d, wts=wt, id=ID.(id))

jk = jackknife(m);
bs = bootstrap(MersenneTwister(12321), m, 1000);
N = nobs(m)
#comparing estimate with jackknife estimate with bootstrap mean
hcat(coef(m), mean(jk, dims=1)[1,:], mean(bs, dims=1)[1,:])
semb = stderror(m)
sebs = std(bs, dims=1)
sejk = std(jk, dims=1, corrected=false) .* sqrt(N-1)
sero = stderror(m, type="robust")

jackknife_vcov(m)
LSurvival.robust_vcov(m)

hcat(semb, sebs[1,:], sejk[1,:], sero)

dat1 = (time = [1, 1, 6, 6, 8, 9], status = [1, 0, 1, 1, 0, 1], x = [1, 1, 1, 0, 0, 0])
dat1clust = (
    id = [1, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    enter = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    exit = [1, 1, 1, 6, 1, 6, 1, 8, 1, 9],
    status = [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    x = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
)

m = coxph(@formula(Surv(time, status)~x),dat1)
mc = coxph(@formula(Surv(enter, exit, status)~x),dat1clust, id=ID.(dat1clust.id))
jk = jackknife(m);
jkc = jackknife(mc);
bs = bootstrap(mc, 100);
std(bs[:,1])
std(jkc[:,1])
stderror(mc)
@assert jk == jkc

"""
function jackknife(m::M) where {M<:PHModel}
    uid = unique(m.R.id)
    coefs = zeros(length(uid), length(coef(m)))
    R = deepcopy(m.R)
    P = deepcopy(m.P)
    for i in eachindex(uid)
        Ri, Rj, idxi, idxj = pop(m.R);
        Pi = popat!(m.P, idxi, idxj)
        mi= PHModel(Rj, m.P, m.formula, m.ties, false, m.bh[1:length(Rj.eventtimes),:], nothing)
        fit!(mi, getbasehaz=false)
        coefs[i,:] = coef(mi)
        m.R = push(Ri, Rj)
        push!(Pi, m.P)
    end
    m.R, m.P = R, P
    coefs
end


function jackknife_vcov(m::M) where {M<:PHModel}
    N = nobs(m)
    #comparing estimate with jackknife estimate with bootstrap mean
    jk = jackknife(m);
    covjk = cov(jk) .* (N-1)
    covjk
end

