using Test
using Anneal

const ISING_SOLVERS = [

]

function main()
    for optimizer in ISING_SOLVERS
        Anneal.test(optimizer)
    end
end

main() # Here we go!