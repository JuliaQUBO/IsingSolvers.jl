module IsingSolvers

# ~*~ Includes: Greedy Descent ~*~ #
include("solvers/greedy_descent.jl")

export GreedyDescent

# ~*~ Includes: Interger Linear Programming ~*~ #
# include("solvers/ilp.jl")
# 
# export ILP

# ~*~ Includes: Monte Carlo Markov Chain ~*~ #
include("solvers/mcmc_random.jl")

export MCMCRandom

end # module
