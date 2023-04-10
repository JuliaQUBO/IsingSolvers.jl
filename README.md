# IsingSolvers.jl
[![QUBODRIVERS](https://img.shields.io/badge/Powered%20by-QUBODrivers.jl-%20%234063d8)](https://github.com/psrenergy/QUBODrivers.jl)

Ising Model solvers inspired by LANL's [ising-solvers](https://github.com/lanl-ansi/ising-solvers) in Julia with wrappers for JuMP

## Ising Model

$$\begin{array}{rl}
\displaystyle \min_{\mathbf{s}} & \mathbf{s}'\mathbf{J}\\,\mathbf{s} + \mathbf{h}'\mathbf{s}\\
\text{s.t.}                     & \mathbf{s} \in \left\lbrace\pm 1\right\rbrace^{n}
\end{array}$$
