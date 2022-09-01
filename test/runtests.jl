using Test
using Anneal
using IsingSolvers

const ISING_SOLVERS = [
    GreedyDescent.Optimizer,
]

main() = Anneal.test.(ISING_SOLVERS)
main() # Here we go!