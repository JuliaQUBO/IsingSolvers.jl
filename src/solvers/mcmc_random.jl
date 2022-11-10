module MCMCRandom

using Anneal # ~ exports MOI = MathOptInterface
using Random

Anneal.@anew Optimizer begin
    name = "Monte Carlo Markov Chain Sampler"
    sense = :min
    domain = :spin
    attributes = begin
        "max_iter"::Union{Integer,Nothing} = 1_000
        NumberOfReads["num_reads"]::Integer = 1_000
        RandomSeed["seed"]::Union{Integer,Nothing} = nothing
    end
end

function Anneal.sample(sampler::Optimizer{T}) where {T}
    # ~*~ Retrieve Model ~*~ #
    h, J, α, β = Anneal.ising(sampler, Dict, T)

    # ~*~ Retrieve attributes ~*~ #
    n           = MOI.get(sampler, MOI.NumberOfVariables())
    time_limit  = MOI.get(sampler, MOI.TimeLimitSec())
    max_iter    = MOI.get(sampler, MOI.RawOptimizerAttribute("max_iter"))
    num_reads   = MOI.get(sampler, MCMCRandom.NumberOfReads())
    random_seed = MOI.get(sampler, MCMCRandom.RandomSeed())

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
    states =
        let results = @timed sample_states(rng, n, h, J, max_iter, time_limit, num_reads)
            # ~*~ Measure time ~*~ #
            time_data["sampling"] = results.time

            results.value
        end

    samples = Anneal.Sample{T,Int}[]

    for ψ in states
        sample = Anneal.Sample{T}(ψ, α * (Anneal.energy(h, J, ψ) + β))

        push!(samples, sample)
    end

    # ~*~ Gather metadata ~*~ #
    metadata = Dict{String,Any}(
        "time" => time_data,
        "origin" => "Greedy Descent Algorithm"
    )

    return Anneal.SampleSet{T}(samples, metadata)
end

function sample_states(
    rng,
    n::Integer,
    h::Dict{Int,T},
    J::Dict{Tuple{Int,Int},T},
    max_iter::Union{Integer,Nothing},
    time_limit::Union{Float64,Nothing},
    num_reads::Integer,
) where {T}
    return Vector{Int}[sample_state(rng, n, h, J, max_iter, time_limit) for _ = 1:num_reads]
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

    # ~*~ Variables ~*~ #
    ψ = Vector{Int}(undef, n)
    ψ⃰ = Vector{Int}(undef, n)
    λ⃰ = Inf

    # ~*~ Run Algorithm ~*~ #
    start_time = time()
    while !stop((time() - start_time), time_limit, num_iter, max_iter)
        # ~ sample random state
        Random.rand!(rng, ψ, (-1, 1))
        # ~ compute its energy
        λ = Anneal.energy(h, J, ψ)

        if λ < λ⃰
            # ~ update best energy
            λ⃰ = λ
            # ~ update best state
            ψ⃰[:] .= ψ[:]
        end

        num_iter += 1
    end

    return ψ⃰
end

# ~*~ Stop criteria ~*~ #
function stop(
    elapsed_time::Float64,
    time_limit::Float64,
    num_iter::Integer,
    max_iter::Integer,
)
    (elapsed_time > time_limit) || (num_iter > max_iter)
end

function stop(::Float64, ::Nothing, num_iter::Integer, max_iter::Integer)
    num_iter > max_iter
end

function stop(elapsed_time::Float64, time_limit::Float64, ::Integer, ::Nothing)
    elapsed_time > time_limit
end

end # module