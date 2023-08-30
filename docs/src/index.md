```@meta
CurrentModule = LSurvival=
```

# [LSurvival](https://github.com/alexpkeil1/LSurvival.jl).

Survival analysis functions in Julia for time-to-event outcomes that can include:
- Loss-to-follow-up/right censoring
- Late entry/left truncation (not available in the Survival.jl package)
- "Person-period" data structures (not available in the Survival.jl package)
- Observation weights (not available in the Survival.jl package)
- Competing risks (not available in the Survival.jl package)

Capabilities include estimators for
- Kaplan-Meier non-parametric conditional risk functions
- Aalen-Johansen non-parametric cause-specific unconditional risk functions
- Cox proportional hazards model (Efron's or Breslow's methods for ties)

Convenience functions enable:
- Non-parametric bootstrapping, cluster-bootstrapping
- Estimating baseline hazards from a Cox Model
- Estimating cause-specific risk from an exhaustive set of Cox models for competing risk outcomes
- Simple simulation of competing and non-competing survival events

Plans to include:
- cluster robust variance estimation (without bootstrapping)

Report issues [here](https://github.com/alexpkeil1/LSurvival.jl/issues)

## Installation 
```julia
using Pkg; Pkg.add(url = "https://github.com/alexpkeil1/LSurvival.jl")
```

## Index of functions
```@index
```

```@autodocs
Modules = [LSurvival]
```
