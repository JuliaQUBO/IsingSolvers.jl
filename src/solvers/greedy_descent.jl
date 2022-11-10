module GreedyDescent

using Anneal # ~ exports MOI = MathOptInterface
using Random

Anneal.@anew Optimizer begin
    name       = "Greedy Descent Solver"
    sense      = :min
    domain     = :spin
    attributes = begin
        "max_iter"::Union{Integer,Nothing}         = 1_000
        NumberOfReads["num_reads"]::Integer        = 1_000
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
    num_reads   = MOI.get(sampler, GreedyDescent.NumberOfReads())
    random_seed = MOI.get(sampler, GreedyDescent.RandomSeed())

    # ~*~ Input validation ~*~ #
    @assert isnothing(max_iter) || max_iter >= 0
    @assert num_reads >= 0
    @assert isnothing(random_seed) || random_seed >= 0

    if isnothing(time_limit) && isnothing(max_iter)
        error("Both 'time_limit' and 'max_iter' are missing. The algorithm must stop!")
    end

    # ~*~ Timing Information ~*~ #
    time_data = Dict{String,Any}()

    # ~*~ Variables ~*~ #
    ℓ = T[get(h, i, zero(T)) for i = 1:n]
    A = Anneal.adjacency(J)
    Ω = sizehint!(BitSet(), n)
    ψ = zeros(Int, n)
    Δ = zeros(T, n, 2)

    # ~*~ Random Generator ~*~ #
    rng = Random.Xoshiro(random_seed)

    # ~*~ Run algorithm ~*~ #
    states = let results = @timed sample_states(
            rng, n, ℓ, h, J, A, Ω, ψ, Δ,
            max_iter, time_limit, num_reads,
        )

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
        "time"   => time_data,
        "origin" => "Greedy Descent Algorithm"
    )

    return Anneal.SampleSet{T}(samples, metadata)
end

function sample_states(
    rng,
    n::Integer,
    ℓ::Vector{T},
    h::Dict{Int,T},
    J::Dict{Tuple{Int,Int},T},
    A::Dict{Int,Set{Int}},
    Ω::BitSet,
    ψ::Vector{Int},
    Δ::Matrix{T},
    max_iter::Union{Integer,Nothing},
    time_limit::Union{Float64,Nothing},
    num_reads::Integer,
) where {T}
    return Vector{Int}[
        sample_state(rng, n, ℓ, h, J, A, Ω, ψ, Δ, max_iter, time_limit)
        for _ = 1:num_reads
    ]
end

function sample_state(
    rng,
    n::Integer,
    ℓ::Vector{T},
    h::Dict{Int,T},
    J::Dict{Tuple{Int,Int},T},
    A::Dict{Int,Set{Int}},
    Ω::BitSet,
    ψ::Vector{Int},
    Δ::Matrix{T},
    max_iter::Union{Integer,Nothing},
    time_limit::Union{Float64,Nothing},
) where {T}
    # ~*~ Counters ~*~ #
    num_iter = 0

    # ~*~ Optimal values ~*~ #
    λ⃰ = Inf
    ψ⃰ = zeros(Int, n)

    # ~*~ Run Algorithm ~*~ #
    start_time = time()
    while !stop((time() - start_time), time_limit, num_iter, max_iter)
        # ~ state = 0
        fill!(ψ, 0)
        # ~ uncheck all variables
        union!(Ω, 1:n)
        # ~ reset transition costs
        Δ[:, :] .= [-ℓ ℓ]

        # ~ loop through variables
        for i = 1:n
            # ~ count iteration
            num_iter += 1

            # ~ pick unvisited variable
            k = rand(rng, Ω)

            # ~ compute transition energy
            δ̂ = Δ[k, 1]
            # ~ candidates for assignment
            ϕ = Tuple{Int,Int}[]

            for j ∈ Ω
                δ = Δ[i, 1]

                if δ < δ̂
                    δ̂ = δ
                    empty!(ϕ)
                end

                if δ <= δ̂
                    push!(ϕ, (j, ↑))
                end

                δ = Δ[i, 2]

                if δ < δ̂
                    δ̂ = δ
                    empty!(ϕ)
                end

                if δ <= δ̂
                    push!(ϕ, (j, ↓))
                end
            end

            if !isempty(ϕ)
                i, σ = rand(rng, ϕ)

                ψ[i] = σ

                delete!(Ω, i)

                for k ∈ A[i]
                    if k ∈ Ω
                        ψ[k] = ↑
                        Δ[k, 1] = φ(k, ℓ[k], A[k], J, ψ)

                        ψ[k] = ↓
                        Δ[k, 2] = φ(k, ℓ[k], A[k], J, ψ)

                        ψ[k] = 0
                    end
                end
            end

            if stop((time() - start_time), time_limit, num_iter, max_iter)
                break
            end
        end

        # ~ collect unvisited variables
        ω = collect(Ω)
        # ~ fill blanks within state
        ψ[ω] .= rand(rng, (↑, ↓), length(ω))

        # ~ evaluate
        λ = Anneal.energy(h, J, ψ)

        # ~ greedy update
        if λ < λ⃰
            λ⃰ = λ
            ψ⃰[:] .= ψ[:]
        end
    end

    # ~ best state so far
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

# ~*~ Scan the neighborhood ~*~ #
function φ(
    i::Integer,
    ℓ::T,
    A::Set{Int},
    J::Dict{Tuple{Int,Int},T},
    ψ::Vector{Int},
) where {T}
    s = ℓ * ψ[i]

    for j in A
        s += ψ[i] * ψ[j] * if i < j
            get(J, (i, j), zero(T))
        else
            get(J, (j, i), zero(T))
        end
    end

    return s
end

end # module