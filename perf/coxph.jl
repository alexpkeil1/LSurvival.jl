using LSurvival, Random


###################################################################
# Fitting a basic Cox model, Kaplan-Meier curve using preferred functions
###################################################################
id, int, outt, data =
    LSurvival.dgm(MersenneTwister(123123), 500_000, 100; afun = LSurvival.int_0)
data[:, 1] = round.(data[:, 1], digits = 3)
d, X = data[:, 4], data[:, 1:3]
wt = rand(length(d))
wt ./= (sum(wt) / length(wt))

# equivalent methods for unweighted, default efron partial likelihood
subn = findall(id .< 100)
coxph(X[subn,:], int[subn], outt[subn], d[subn])            # trigger compilation
@time coxph(X, int, outt, d, ties="efron");
