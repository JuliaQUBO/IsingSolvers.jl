module MCMCRandom

using Anneal # ~ exports MOI = MathOptInterface
using Random

Anneal.@anew Optimizer begin
    name = "Monte Carlo Markov Chain Sampler"
    sense = :min
    domain = :spin
    attributes = begin
        "max_iter"::Union{Integer,Nothing} = 1_000
        "num_reads"::Integer = 1_000
        "random_seed"::Union{Integer,Nothing} = nothing
    end
end

function Anneal.sample(sampler::Optimizer{T}) where {T}
    # ~*~ Retrieve Model ~*~ #
    h, J = Anneal.ising(Dict, T, sampler)

    # ~*~ Retrieve attributes ~*~ #
    n           = MOI.get(sampler, MOI.NumberOfVariables())
    time_limit  = MOI.get(sampler, MOI.TimeLimitSec())
    max_iter    = MOI.get(sampler, MOI.RawOptimizerAttribute("max_iter"))
    num_reads   = MOI.get(sampler, MOI.RawOptimizerAttribute("num_reads"))
    random_seed = MOI.get(sampler, MOI.RawOptimizerAttribute("random_seed"))

    # ~*~ Input validation ~*~ #
    @assert isnothing(max_iter) || max_iter >= 0
    @assert num_reads >= 0
    @assert isnothing(random_seed) || random_seed >= 0

    if isnothing(time_limit) && isnothing(max_iter)
        error("Both 'time_limit' and 'max_iter' are missing. The algorithm must stop!")
    end

    # ~*~ Timing Information ~*~ #
    time_data = Dict{String,Any}()

    # ~*~ Random Generator ~*~ #
    rng = Random.Xoshiro(random_seed)

    # ~*~ Run algorithm ~*~ #
    result = @timed Vector{Int}[
        sample_state(rng, n, h, J, max_iter, time_limit)
        for _ = 1:num_reads
    ]
    # ~*~ Format Results ~*~ #
    states = result.value

    # ~*~ Read time ~*~ #
    time_data["sampling"] = result.time

    # ~*~ Gather metadata ~*~ #
    metadata = Dict{String,Any}(
        "time" => time_data,
        "origin" => "Greedy Descent Algorithm"
    )

    return Anneal.SampleSet{Int,T}(sampler, states, metadata)
end

function sample_state(
    rng,
    n::Integer,
    h::Dict{Int,T},
    J::Dict{Tuple{Int,Int},T},
    max_iter::Union{Integer,Nothing},
    time_limit::Union{Float64,Nothing},
) where {T}
    # ~*~ Counters ~*~ #
    num_iter = 0

    # ~*~ Optimal values ~*~ #
    ψ⃰ = rand(rng, (-1, 1), n)
    λ⃰ = Anneal.energy(ψ⃰, h, J)

    # ~*~ Run Algorithm ~*~ #
    start_time = time()
    while !stop((time() - start_time), time_limit, num_iter, max_iter)
        ψ = rand(rng, (-1, 1), n)
        λ = Anneal.energy(ψ, h, J)

        if λ < λ⃰
            λ⃰ = λ
            ψ⃰[:] .= ψ[:]
        end

        num_iter += 1
    end

    return ψ⃰
end

# ~*~ Stop criteria ~*~ #
function stop(elapsed_time::Float64, time_limit::Float64, num_iter::Integer, max_iter::Integer)
    (elapsed_time > time_limit) || (num_iter > max_iter)
end

function stop(::Float64, ::Nothing, num_iter::Integer, max_iter::Integer)
    num_iter > max_iter
end

function stop(elapsed_time::Float64, time_limit::Float64, ::Integer, ::Nothing)
    elapsed_time > time_limit
end

end # module