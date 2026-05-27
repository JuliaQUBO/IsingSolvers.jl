# IsingSolvers.jl
[![QUBODRIVERS](https://img.shields.io/badge/Powered%20by-QUBODrivers.jl-%20%234063d8)](https://github.com/JuliaQUBO/QUBODrivers.jl)

## Legacy Notice

`IsingSolvers.jl` is a legacy package and is no longer recommended for new work.
This repository is kept available for compatibility and historical reference, but
the maintained QUBO solver interfaces now live in the broader
[QUBODrivers.jl](https://github.com/JuliaQUBO/QUBODrivers.jl) ecosystem.

Recommended replacements:

| Legacy solver | Recommended replacement |
| :-- | :-- |
| `IsingSolvers.MCMCRandom` | `QUBODrivers.RandomSampler` |
| `IsingSolvers.GreedyDescent` | `DWave.Greedy` or `DWave.Neal` |
| `IsingSolvers.ILP` | A forthcoming MOI-native `QUBODrivers.MIPSampler` baseline |

This package was originally inspired by LANL's
[ising-solvers](https://github.com/lanl-ansi/ising-solvers) reference code. That
repository is mentioned here only as historical provenance.

## Overview

Legacy Ising model solvers in Julia with wrappers for JuMP.

## Ising Model

$$\begin{array}{rl}
\displaystyle \min_{\mathbf{s}} & \mathbf{s}'\mathbf{J}\\,\mathbf{s} + \mathbf{h}'\mathbf{s}\\
\text{s.t.}                     & \mathbf{s} \in \left\lbrace\pm 1\right\rbrace^{n}
\end{array}$$
