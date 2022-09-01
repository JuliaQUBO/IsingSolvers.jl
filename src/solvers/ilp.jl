module ILP

# ~*~ Imports: Anneal.jl ~*~ #
using Anneal

# ~*~ Imports: MIP Solvers ~*~ #
using JuMP
using HiGHS
using Gurobi

Anneal.@anew Optimizer begin
    name = "ILP"
    sense = :min
    domain = :spin
    attributes = begin
        "cuts"::Union{Integer,Nothing} = nothing
        "mip_solver"::MOI.AbstractOptimizer = Gurobi.Optimizer
    end
end

function Anneal.sample(sampler::Optimizer{T}) where {T}
    n = MOI.get(sampler, MOI.NumberOfVariables())
    Q, = Anneal.qubo(Dict, T, sampler)
    h, J = Anneal.ising(Dict, T, sampler)

    mip_solver = MOI.get(sampler, MOI.RawOptimizerAttribute("mip_solver"))
    mip_model = Model(mip_solver)

    cuts = MOI.get(sampler, MOI.RawOptimizerAttribute("cuts"))
    time_limit = MOI.get(sampler, MOI.TimeLimitSec())
    num_threads = MOI.get(sampler, MOI.NumberOfThreads())

    mip_config!(
        mip_model,
        mip_solver;
        cuts=cuts,
        time_limit=time_limit,
        num_threads=num_threads
    )

    cache = Dict()

    @variable(model, x[i=1:n], Bin)


    @variable(model, y[keys(J)], Bin)

    for (i, j) in keys(J)
        @constraint(model, y[(i, j)] >= x[i] + x[j] - 1)
        @constraint(model, y[(i, j)] <= x[i])
        @constraint(model, y[(i, j)] <= x[j])
    end

    if length(h) == 0
        @warn "Spin symmetry detected. Adding symmetry breaking constraint."
        @constraint(model, x[i] == 0)
    end

    @objective(
        model,
        Min,
        sum(c * y[(i, j)] for ((i, j), c) in J) +
        sum(c * x[i] for (i, c) in h)
    )

end

function mip_config!(::Any, ::Type{<:MOI.AbstractOptimizer}; kws...) end

function mip_config!(model, ::Type{<:Gurobi.Optimizer};
    cuts::Union{Integer,Nothing}=nothing,
    num_threads::Integer=1,
    time_limit::Union{Float64,Nothing}=nothing,
    kws...)
    set_optimizer_attribute(model, "Presolve", 2)
    set_optimizer_attribute(model, "MIPFocus", 1)

    set_optimizer_attribute(model, MOI.NumberOfThreads(), num_threads)
    set_optimizer_attribute(model, MOI.TimeLimitSec(), time_limit)

    if !isnothing(cuts)
        set_optimizer_attribute(model, "Cuts", cuts)
    end
end

end