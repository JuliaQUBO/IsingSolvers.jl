module ILP

# ~*~ Imports: Anneal.jl ~*~ #
using Anneal

# ~*~ Imports: MIP Solvers ~*~ #
using JuMP
using HiGHS
using Gurobi

Anneal.@anew Optimizer begin
    name       = "ILP"
    sense      = :min
    domain     = :bool
    attributes = begin
        "mip_cuts"::Union{Integer,Nothing} = nothing
        "mip_solver"::Any = HiGHS.Optimizer
    end
end

function Anneal.sample(sampler::Optimizer{T}) where {T}
    n    = MOI.get(sampler, MOI.NumberOfVariables())
    Q,   = Anneal.qubo(Dict, T, sampler)
    h, J = Anneal.ising(Dict, T, sampler)

    mip_solver = MOI.get(sampler, MOI.RawOptimizerAttribute("mip_solver"))
    mip_cuts   = MOI.get(sampler, MOI.RawOptimizerAttribute("mip_cuts"))

    time_limit  = MOI.get(sampler, MOI.TimeLimitSec())
    num_threads = MOI.get(sampler, MOI.NumberOfThreads())
    silent      = MOI.get(sampler, MOI.Silent())

    # ~*~ Timing Information ~*~ #
    time_data = Dict{String,Any}()

    # ~*~ Build ILP Model ~*~ #
    model = Model(mip_solver)

    result = @timed begin
        mip_config!(
            model,
            mip_solver;
            cuts        = mip_cuts,
            time_limit  = time_limit,
            num_threads = num_threads,
            silent      = silent,
        )

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

        model
    end

    # ~*~ Read time ~*~ #
    time_data["build_model"] = result.time

    # ~*~ Run ILP Optimization ~*~ #
    result = @timed begin
        optimize!(model)

        trunc.(Int, value.(model[:x]))
    end

    # ~*~ Format Results ~*~ #
    state = result.value

    # ~*~ Read time ~*~ #
    time_data["optimization"] = result.time

    # ~*~ Gather metadata ~*~ #
    metadata =
        Dict{String,Any}("time"   => time_data, "origin" => "ILP @ $(solver_name(model))")

    return Anneal.SampleSet{Int,T}(sampler, [state], metadata)
end

function mip_config!(::Any, ::Type{<:MOI.AbstractOptimizer}; kws...)
    nothing
end

function mip_config!(
    model,
    ::Type{<:Gurobi.Optimizer};
    cuts::Union{Integer,Nothing} = nothing,
    num_threads::Integer = 1,
    time_limit::Union{Float64,Nothing} = nothing,
    silent::Bool = false,
    kws...,
)
    # set_optimizer_attribute(model, "Presolve", 2)
    # set_optimizer_attribute(model, "MIPFocus", 1)

    set_optimizer_attribute(model, MOI.NumberOfThreads(), num_threads)
    set_optimizer_attribute(model, MOI.TimeLimitSec(), time_limit)
    set_optimizer_attribute(model, MOI.Silent(), silent)

    if !isnothing(cuts)
        set_optimizer_attribute(model, "Cuts", cuts)
    end
end

function mip_config!(
    model,
    ::Type{<:HiGHS.Optimizer};
    num_threads::Integer = 1,
    time_limit::Union{Float64,Nothing} = nothing,
    silent::Bool = false,
    kws...,
)
    set_optimizer_attribute(model, MOI.NumberOfThreads(), num_threads)
    set_optimizer_attribute(model, MOI.TimeLimitSec(), time_limit)
    set_optimizer_attribute(model, MOI.Silent(), silent)
end

end # module