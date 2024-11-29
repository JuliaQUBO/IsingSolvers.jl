module ILP

using JuMP
import QUBODrivers:
    MOI,
    QUBODrivers,
    QUBOTools,
    Sample,
    SampleSet,
    @setup,
    sample

@setup Optimizer begin
    name       = "ILP"
    attributes = begin
        MIPSolver["mip_solver"]::Any = nothing
    end
end

function sample(sampler::Optimizer{T}) where {T}
    # Retrieve Model
    n, L, Q, α, β = QUBOTools.qubo(sampler, :dict; sense = :min)

    # Retrieve Attributes
    solver = MOI.get(sampler, ILP.MIPSolver())
    params = Dict{Symbol,Any}(
        :time_limit  => MOI.get(sampler, MOI.TimeLimitSec()),
        :num_threads => MOI.get(sampler, MOI.NumberOfThreads()),
        :silent      => MOI.get(sampler, MOI.Silent()),
    )

    @assert !isnothing(solver)

    # Build ILP Model
    results = @timed build_mip_model(solver, n, L, Q; params...)
    model   = results.value

    metadata = Dict{String,Any}(
        "origin" => "$(JuMP.solver_name(model)) @ IsingSolvers.jl",
        "time"   => Dict{String,Any}()
    )

    # Measure time
    metadata["time"]["build_model"] = results.time

    # Run ILP Optimization
    results = @timed solve_mip_model(model)
    states  = [results.value]

    samples = Sample{T,Int}[]

    for ψ in states
        λ = QUBOTools.value(ψ, L, Q, α, β)
    
        push!(samples, Sample{T}(ψ, λ))
    end
    
    # Measure time
    metadata["time"]["effective"] = results.time

    return SampleSet{T}(samples, metadata; domain = :bool, sense = :min)
end

function build_mip_model(
    solver,
    n::Integer,
    L::Dict{Int,T},
    Q::Dict{Tuple{Int,Int},T};
    kws...,
) where {T}
    model = JuMP.Model(solver)

    JuMP.@variable(model, x[i = 1:n], Bin)
    JuMP.@variable(model, y[keys(Q)], Bin)

    for (i, j) in keys(Q)
        JuMP.@constraint(model, y[(i, j)] >= x[i] + x[j] - 1)
        JuMP.@constraint(model, y[(i, j)] <= x[i])
        JuMP.@constraint(model, y[(i, j)] <= x[j])
    end

    # if all(iszero.(values(h)))
    #     @warn "Spin symmetry detected. Adding symmetry breaking constraint."
    #     JuMP.@constraint(model, x[1] == 0)
    # end

    JuMP.@objective(
        model,
        Min,
        sum(c * x[j] for (j, c) in L) + sum(c * y[(i, j)] for ((i, j), c) in Q)
    )

    return model
end

function solve_mip_model(model)
    optimize!(model)

    return round.(Int, value.(model[:x]))
end

end # module