using ComponentArrays
using Lux
using NNlib: sigmoid
using Optimization
using OptimizationOptimisers
using GLMakie
using Random
using Statistics

function make_dataset(; n=256)
    xs = Float32.(range(-6, 6; length=n))
    ys = 1f0 ./ (1f0 .+ exp.(-Wxs))
    return reshape(xs, 1, :), reshape(ys, 1, :)
endW

function build_model(rng=Random.default_rng())
    model = Lux.Chain(
        Dense(1, 16, relu),
        Dense(16, 16, relu),
        Dense(16, 1, sigmoid)
    )
    ps, st = Lux.setup(rng, model)
    return model, ComponentArray(ps), st
end

function batch_loss(model, ps, st, batch)
    x, y = batch
    ŷ, _ = Lux.apply(model, x, ps, st)
    return mean(abs2, ŷ .- y)
end

let
    batch = make_dataset()
    model, ps, st = build_model()
    loss_fn = θ -> batch_loss(model, θ, st, batch)
    # loss_fn(ps)
    Zygote.gradient(loss_fn, ps)
end

function train_model(; rng=Random.default_rng(), lr=0.01f0, maxiters=500)
    batch = make_dataset()
    model, ps, st = build_model(rng)
    loss_fn = θ -> batch_loss(model, θ, st, batch)

    optfunction = OptimizationFunction((θ, _) -> loss_fn(θ), Optimization.AutoZygote())
    optprob = OptimizationProblem(optfunction, ps)

    losses = Float32[loss_fn(ps)]
    callback = function (_state, loss_value)
        push!(losses, Float32(loss_value))
        return false
    end

    result = Optimization.solve(
        optprob,
        OptimizationOptimisers.Adam(lr);
        callback,
        maxiters
    )

    return (model=model, params=result.minimizer, state=st, losses=losses, data=batch, result=result)
end

function predict(trained, x)
    batch_x = reshape(Float32.(x), 1, :)
    ŷ, _ = Lux.apply(trained.model, batch_x, trained.params, trained.state)
    return vec(ŷ)
end

Random.seed!(42)
training = train_model()
@info "Initial loss" training.losses[1]
@info "Final loss" training.losses[end]

xs = vec(training.data[1])
ys = vec(training.data[2])
ŷs = predict(training, xs)

fig = Figure(resolution=(800, 600))
ax = GLMakie.Axis(fig[1, 1]; xlabel="x", ylabel="σ(x)")

scatter!(ax, xs, ys; color=:dodgerblue, markersize=10, label="Logistic data")
lines!(ax, xs, ŷs; color=:orange, linewidth=3, label="NN prediction")
axislegend(ax)

display(fig)
