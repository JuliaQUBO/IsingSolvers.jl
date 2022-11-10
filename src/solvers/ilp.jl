module ILP

# ~*~ Imports: Anneal.jl ~*~ #
using Anneal

# ~*~ Imports: MIP Solvers ~*~ #
using JuMP
using HiGHS

Anneal.@anew Optimizer begin
    name       = "ILP"
    sense      = :min
    domain     = :bool
    attributes = begin
        MIPSolver["mip_solver"]::Any = HiGHS.Optimizer
    end
end

function Anneal.sample(sampler::Optimizer{T}) where {T}
    # ~*~ Retrieve Model ~*~ #
    n          = MOI.get(sampler, MOI.NumberOfVariables())
    Q, α, β    = Anneal.qubo(sampler, Dict, T)
    h, J, _, _ = Anneal.ising(sampler, Dict, T)

    # ~*~ Retrieve Attributes ~*~ #
    solver = MOI.get(sampler, ILP.MIPSolver())
    params = Dict{Symbol,Any}(
        :time_limit  => MOI.get(sampler, MOI.TimeLimitSec()),
        :num_threads => MOI.get(sampler, MOI.NumberOfThreads()),
        :silent      => MOI.get(sampler, MOI.Silent()),
    )

    # ~*~ Timing Information ~*~ #
    time_data = Dict{String,Any}()

    # ~*~ Build ILP Model ~*~ #

    results = @timed build_mip_model(solver, n, Q, h, J; params...)
    model   = results.value

    # ~*~ Measure time ~*~ #
    time_data["build_model"] = results.time

    # ~*~ Run ILP Optimization ~*~ #
    results = @timed solve_mip_model(model)
    states  = [results.value]

    samples = Anneal.Sample{T,Int}[]

    for ψ in states
        sample = Anneal.Sample{T}(ψ, α * (Anneal.energy(Q, ψ) + β))

        push!(samples, sample)
    end
    
    # ~*~ Measure time ~*~ #
    time_data["solve"] = results.time

    # ~*~ Gather metadata ~*~ #
    metadata = Dict{String,Any}(
        "time"   => time_data,
        "origin" => "ILP @ $(JuMP.solver_name(model))",
    )

    return Anneal.SampleSet{T}(samples, metadata)
end

function build_mip_model(
    solver,
    n::Integer,
    Q::Dict{Tuple{Int,Int},T},
    h::Dict{Int,T},
    J::Dict{Tuple{Int,Int},T};
    kws...,
) where {T}
    model = JuMP.Model(solver)

    JuMP.@variable(model, x[i = 1:n], Bin)
    JuMP.@variable(model, y[keys(J)], Bin)

    for (i, j) in keys(J)
        JuMP.@constraint(model, y[(i, j)] >= x[i] + x[j] - 1)
        JuMP.@constraint(model, y[(i, j)] <= x[i])
        JuMP.@constraint(model, y[(i, j)] <= x[j])
    end

    if all(iszero.(values(h)))
        @warn "Spin symmetry detected. Adding symmetry breaking constraint."
        JuMP.@constraint(model, x[begin] == 0)
    end

    JuMP.@objective(
        model,
        Min,
        sum((i == j ? c * x[i] : c * y[(i, j)]) for ((i, j), c) in Q)
    )

    mip_config!(model, solver; kws...)

    return model
end

function solve_mip_model(model)
    optimize!(model)

    return round.(Int, value.(model[:x]))
end

function mip_config!(::Any, ::Any; kws...) end

# function mip_config!(
#     model,
#     ::Type{<:Gurobi.Optimizer};
#     cuts::Union{Integer,Nothing} = nothing,
#     num_threads::Integer = 1,
#     time_limit::Union{Float64,Nothing} = nothing,
#     silent::Bool = false,
#     kws...,
# )
#     # set_optimizer_attribute(model, "Presolve", 2)
#     # set_optimizer_attribute(model, "MIPFocus", 1)
#     time_limit = isnothing(time_limit) ? Inf : time_limit

#     set_optimizer_attribute(model, MOI.NumberOfThreads(), num_threads)
#     set_optimizer_attribute(model, MOI.TimeLimitSec(), time_limit)
#     set_optimizer_attribute(model, MOI.Silent(), silent)

#     if !isnothing(cuts)
#         set_optimizer_attribute(model, "Cuts", cuts)
#     end

#     set_optimizer_attribute(model, "Presolve", 0)
# end

function mip_config!(
    model,
    ::Type{O};
    num_threads::Integer = 1,
    time_limit::Union{Float64,Nothing} = nothing,
    silent::Bool = false,
    kws...,
) where {O<:HiGHS.Optimizer}
    set_optimizer_attribute(model, MOI.NumberOfThreads(), num_threads)
    set_optimizer_attribute(model, MOI.TimeLimitSec(), time_limit)
    set_optimizer_attribute(model, MOI.Silent(), true)
end

end # module