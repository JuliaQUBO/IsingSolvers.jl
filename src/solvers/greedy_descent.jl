module GreedyDescent

using Anneal # ~ exports MOI = MathOptInterface

Anneal.@anew Optimizer begin
    name = "Greedy Descent Solver"
    sense = :min
    domain = :spin
    attributes = begin
        "random_seed"::Union{Integer,Nothing} = nothing
    end
end

function Anneal.sample(sampler::Optimizer{T}) where {T}
    # ~*~ Retrieve Model ~*~ #
    h, J = Anneal.ising(Dict, T, sampler)

    # ~*~ Retrieve attributes ~*~ #
    n = MOI.get(sampler, MOI.NumberOfVariables())
    time_limit = MOI.get(sampler, MOI.TimeLimitSec())

    # ~*~ Timing Information ~*~ #
    time_data = Dict{String,Any}()

    # ~*~ Variables ~*~ #
    ℓ = T[get(h, i, zero(T)) for i = 1:n]
    ψ = zeros(Int, n)
    Δ = zeros(Int, n, 2)
    ψ⃰ = zeros(Int, n)
    λ⃰ = Inf
    A = Anneal.adjacency(J)
    Ω = sizehint!(BitSet(), n)

    # ~*~ Run algorithm ~*~ #
    num_iter = 0
    restarts = 0

    result = @timed begin
        start_time = time()
        while (time() - start_time) < time_limit
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
                k = rand(Ω)

                # ~ compute transition energy
                δ̂ = Δ[k, 1]
                # ~ candidates for assignment
                ϕ = Tuple{Int,Int}[]

                for j ∈ Ω
                    δ = Δ[i, 1]
                    if δ < δ̂
                        δ̂ = δ
                        empty!(ϕ)
                    else
                        push!(ϕ, Tuple{Int,Int,Float64}(j, -1))
                    end

                    δ = Δ[i, 2]
                    if δ < δ̂
                        δ̂ = δ
                        empty!(ϕ)
                    else
                        push!(ϕ, Tuple{Int,Int,Float64}(j, 1))
                    end
                end

                i, σ = rand(ϕ)

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

                if (time() - start_time) > time_limit
                    break
                end
            end

            # ~ collect unvisited variables
            ω = collect(Ω)
            # ~ fill blanks within state
            ψ[ω] .= rand((-1, 1), length(ω))

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
        ψ⃰
    end
    # ~*~ Read time ~*~ #
    time_data["sampling"] = result.time

    # ~*~ Format Results ~*~ #
    state = result.value

    # ~*~ Gather metadata ~*~ #
    metadata = Dict{String,Any}(
        "time" => time_data,
        "origin" => "Greedy Descent Algorithm"
    )

    return Anneal.SampleSet{U,T}(sampler, [state], metadata)
end

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