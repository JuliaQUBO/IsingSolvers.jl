using IsingSolvers: MOI, QUBODrivers, IsingSolvers
using GLPK

QUBODrivers.test(IsingSolvers.GreedyDescent.Optimizer)
QUBODrivers.test(IsingSolvers.MCMCRandom.Optimizer)
QUBODrivers.test(IsingSolvers.ILP.Optimizer) do model
    MOI.set(model, IsingSolvers.ILP.MIPSolver(), GLPK.Optimizer)
end