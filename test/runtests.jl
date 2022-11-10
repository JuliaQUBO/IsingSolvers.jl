using Test
using Anneal
using IsingSolvers

GreedyDescent.test(; examples=true)
MCMCRandom.test(; examples=true)
ILP.test(; examples=true)