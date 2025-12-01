using DifferentialEquations
using GLMakie
using Lux
using Lux.Training: TrainState, single_train_step, AutoZygote
using Optimisers
using OrdinaryDiffEq
using Random
using Statistics
using Zygote
using Optimization
using OptimizationOptimisers
using ComponentArrays
using Optim
using OptimizationOptimJL
using DiffEqFlux

function logistic_deriv!(du, u, p, t)
    r, K = p
    du[1] = r * u[1] * (1 - u[1] / K)
end

logistic_solution(t, u0, p) = begin
    r, K = p
    K / (1 + ((K / u0) - 1) * exp(-r * t))
end

u0 = Float32[0.025f0]
tspan = (0f0, 60f0)
p = (0.1f0, 1f0)
prob = ODEProblem(logistic_deriv!, u0, tspan, p)

sol = solve(prob, Tsit5(); saveat=0.01f0)

# Generate (t, u(t)) pairs of the logistic function for training a model.
training_times = collect(range(tspan[1], tspan[2]; length=256))
training_samples = [(τ, logistic_solution(τ, u0[1], p)) for τ in training_times]
sample_t = Float32.(first.(training_samples))
sample_u = Float32.(last.(training_samples))


# Prepare batched training data (features × batch size).
input_batch = reshape(sample_t, 1, :)
target_batch = reshape(sample_u, 1, :)

let
    GLMakie.activate!(inline=false)
    sol_array = Array(sol)
    t_vals = sol.t
    population = sol_array[1, :]

    fig = Figure()
    ax = GLMakie.Axis(fig[1, 1], xlabel="t", ylabel="u(t)", title="Logistic Growth ODE")
    lines!(ax, t_vals, population, color=:steelblue, label="Logistic solution")
    scatter!(ax, sample_t, sample_u, color=:orange, markersize=6, label="Training samples")

    axislegend(ax, position=:rb)

    fig[0, :] = Label(fig, "1D Logistic Growth Integration", fontsize=22)
    display(fig)
end

function neuralode(ps, ts)
    dudt(u, p, t) = begin
        # Treat the scalar state as a 1×1 batch so Lux.apply matches Dense expectations.
        y, _ = Lux.apply(model, reshape(u, :, 1), p, st)
        return vec(y)
    end
    basic_tgrad(u, p, t) = zero(u)
    ff = ODEFunction{false}(dudt; tgrad=basic_tgrad)
    prob = ODEProblem(ff, u0, tspan, ps)

    # sol = solve(prob, Tsit5(); saveat = ts)
    sol = solve(prob, Tsit5(); saveat=ts,
        sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()))
    return sol
end
function loss(ps)
    sol = neuralode(ps, sample_t)
    pred_u = Array(sol)[1, :]
    return mean((pred_u .- sample_u) .^ 2)
end

model = Lux.Chain(
    Dense(1, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 1, sigmoid)
)
ps_nt, st = Lux.setup(Random.default_rng(), model)
ps = ComponentArray(ps_nt)

loss(ps)
Zygote.gradient(loss, ps)

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
pinit = ComponentArray(ps)
optprob = Optimization.OptimizationProblem(optf, pinit)



callback = function (state, l)
    println(l)
    return false
end

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.005); callback=callback, maxiters=1000
)

optprob2 = remake(optprob; u0=result_neuralode.u)


result_neuralode2 = Optimization.solve(
    optprob2, Optim.BFGS(; initial_stepnorm=0.01); callback, allow_f_increases=false)


let
    sol = neuralode(result_neuralode.u, sample_t)

    f = Figure()
    ax = GLMakie.Axis(f[1, 1], xlabel="t", ylabel="u(t)", title="Neural ODE Logistic Growth")
    sol_array = Array(sol)
    t = sol.t
    u = sol_array[1, :]
    lines!(ax, t, u, color=:green, label="Neural
ODE solution")
    scatter!(ax, sample_t, sample_u, color=:orange, markersize=6, label="Training samples")
    axislegend(ax, position=:rb)
    f[0, :] = Label(f, "Neural ODE Logistic Growth Integration", fontsize=22)
    display(f)
end
