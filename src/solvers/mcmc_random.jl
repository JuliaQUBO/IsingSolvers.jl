module MCMCRandom

using Random
import QUBODrivers:
    MOI,
    QUBODrivers,
    QUBOTools,
    Sample,
    SampleSet,
    @setup,
    sample

@setup Optimizer begin
    name       = "Monte Carlo Markov Chain Sampler"
    attributes = begin
        "max_iter"::Union{Integer,Nothing} = 1_000
        NumberOfReads["num_reads"]::Integer = 1_000
        RandomSeed["seed"]::Union{Integer,Nothing} = nothing
    end
end

function sample(sampler::Optimizer{T}) where {T}
    # Retrieve Model
    n, h, J, α, β = QUBOTools.ising(sampler, :dict; sense = :min)

    # Retrieve attributes
    time_limit  = MOI.get(sampler, MOI.TimeLimitSec())
    max_iter    = MOI.get(sampler, MOI.RawOptimizerAttribute("max_iter"))
    num_reads   = MOI.get(sampler, MCMCRandom.NumberOfReads())
    random_seed = MOI.get(sampler, MCMCRandom.RandomSeed())

    # Input validation
    @assert isnothing(max_iter) || max_iter >= 0
    @assert num_reads >= 0
    @assert isnothing(random_seed) || random_seed >= 0

    if isnothing(time_limit) && isnothing(max_iter)
        error("Both 'time_limit' and 'max_iter' are missing. The algorithm must stop!")
    end

    metadata = Dict{String,Any}(
        "origin" => "MCMC Random @ IsingSolvers.jl",
        "time"   => Dict{String,Any}(),
    )

    # Random Generator
    rng = Random.Xoshiro(random_seed)

    # Run algorithm
    results = @timed sample_states(rng, n, h, J, max_iter, time_limit, num_reads)

    metadata["time"]["effective"] = results.time

    samples = Sample{T,Int}[]

    for ψ in results.value
        λ = QUBOTools.value(ψ, h, J, α, β)

        push!(samples, Sample{T}(ψ, λ))
    end

    return SampleSet{T}(samples, metadata)
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
    # Counters
    num_iter = 0

    # Variables
    ψ = Vector{Int}(undef, n)
    ψ⃰ = Vector{Int}(undef, n)
    λ⃰ = Inf

    # Run Algorithm
    start_time = time()
    while !stop((time() - start_time), time_limit, num_iter, max_iter)
        # sample random state
        Random.rand!(rng, ψ, (-1, 1))
        # compute its energy
        λ = QUBOTools.value(h, J, ψ)

        if λ < λ⃰
            # update best energy
            λ⃰ = λ
            # update best state
            ψ⃰ .= ψ
        end

        num_iter += 1
    end

    return ψ⃰
end

# Stop criteria
function stop(
    elapsed_time::Float64,
    time_limit::Float64,
    num_iter::Integer,
    max_iter::Integer,
)
    return (elapsed_time > time_limit) || (num_iter > max_iter)
end

function stop(::Float64, ::Nothing, num_iter::Integer, max_iter::Integer)
    return num_iter > max_iter
end

function stop(elapsed_time::Float64, time_limit::Float64, ::Integer, ::Nothing)
    return elapsed_time > time_limit
end

end # module