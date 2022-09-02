module GreedyDescent

using Anneal # ~ exports MOI = MathOptInterface
using Random

Anneal.@anew Optimizer begin
    name = "Greedy Descent Solver"
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

    # ~*~ Variables ~*~ #
    ℓ = T[get(h, i, zero(T)) for i = 1:n]
    A = Anneal.adjacency(J)
    Ω = sizehint!(BitSet(), n)
    ψ = zeros(Int, n)
    Δ = zeros(T, n, 2)

    # ~*~ Random Generator ~*~ #
    rng = Random.Xoshiro(random_seed)

    # ~*~ Run algorithm ~*~ #
    result = @timed Vector{Int}[
        sample_state(rng, n, ℓ, h, J, A, Ω, ψ, Δ, max_iter, time_limit)
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
    ℓ::Vector{T},
    h::Dict{Int,T},
    J::Dict{Tuple{Int,Int},T},
    A::Dict{Int,Set{Int}}, # adjacency list
    Ω::BitSet,
    ψ::Vector{Int},
    Δ::Matrix{T},
    max_iter::Union{Integer,Nothing},
    time_limit::Union{Float64,Nothing},
) where {T}
    # ~*~ Counters ~*~ #
    num_iter = 0
    restarts = 0
    
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
                    push!(ϕ, (j, -1))
                end

                δ = Δ[i, 2]

                if δ < δ̂
                    δ̂ = δ
                    empty!(ϕ)
                end
                
                if δ <= δ̂
                    push!(ϕ, (j, 1))
                end
            end

            i, σ = rand(rng, ϕ)

            ψ[i] = σ

            delete!(Ω, i)

            for k ∈ A[i]
                if k ∈ Ω
                    ψ[k] = -1
                    Δ[k, 1] = φ(k, ℓ[k], A[k], J, ψ)

                    ψ[k] = 1
                    Δ[k, 2] = φ(k, ℓ[k], A[k], J, ψ)

                    ψ[k] = 0
                end
            end

            if stop((time() - start_time), time_limit, num_iter, max_iter)
                break
            end
        end

        # ~ collect unvisited variables
        ω = collect(Ω)
        # ~ fill blanks within state
        ψ[ω] .= rand(rng, (-1, 1), length(ω))

        # ~ evaluate
        λ = Anneal.energy(ψ, h, J)

        # ~ greedy update
        if λ < λ⃰
            λ⃰ = λ
            ψ⃰[:] .= ψ[:]
        end

        # ~ noch einmal
        restarts += 1
    end

    # ~ best state so far
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

# ~*~ Scan the neighborhood ~*~ #
function φ(i::Integer, ℓ::T, A::Set{Int}, J::Dict{Tuple{Int,Int},T}, ψ::Vector{Int}) where {T}
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