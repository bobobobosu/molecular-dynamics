import ForwardDiff
using GLMakie
using StaticArrays
using LinearAlgebra
using StatsBase
using Tullio
using DifferentiationInterface
using ProgressMeter
GLMakie.activate!(inline=false)
β = 1.0
# Two Gaussian wells with a small central bump; flat (~0) away from the wells.
function double_well(x)
    well_width = 0.6
    well_depth = 1.0
    barrier_height = 0.25

    gauss(cx, cy) = exp(-((x[1] - cx)^2 + (x[2] - cy)^2) / (2 * well_width^2))
    wells = -well_depth * (gauss(-1, -1) + gauss(1, 1))
    # central_bump = barrier_height * exp(-(x[1]^2 + x[2]^2) / (2 * (0.5 * well_width)^2))
    return wells # + central_bump
end

function potential_energy(x)
    return ((x[1]^2 / 4) - 1)^2 + 0.5 * (x[2] + 0.5 * x[1])^2
end

function energy_and_force(x)
    energy, force = value_and_gradient(potential_energy, AutoForwardDiff(), x)
    return energy, -force + rand(MvNormal([0.0, 0.0], 1.0))
end

function energy_and_force(x, bias_potential::Function)
    f(x) = potential_energy(x) + bias_potential(cv(x))
    energy, force = value_and_gradient(f, AutoForwardDiff(), x)


    # fb(x) = bias_potential(cv(x))
    # biased_force = ForwardDiff.gradient(fb, collect(x))
    # biased_energy = fb(collect(x))
    # if rand() < 0.1
    #     println("bias_potential at cv: ", biased_energy, " biased_force: ", biased_force)
    # end

    return energy, -force# + rand(MvNormal([0.0, 0.0], 1.0))
end

using Distributions
rand(MvNormal([0.0, 0.0], 1.0))

function make_grid(f; xlim=(-2.5, 2.5), ylim=(-2.0, 2.0), n=200)
    xs = range(xlim...; length=n)
    ys = range(ylim...; length=n)
    z = Array{Float64}(undef, n, n)
    @tullio z[i, j] := f(SVector{2,Float64}(xs[i], ys[j]))
    return xs, ys, z
end

let
    xs, ys, z = make_grid(potential_energy; xlim=(-3, 3), ylim=(-3, 3))
    vec(sum(z; dims=2))
end

function plot_potential(xs, ys, z)

    fig = Figure(resolution=(1500, 420))

    ax3d = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="V(x, y)",
        title="Flat double-well surface", aspect=:data)
    surface!(ax3d, xs, ys, z; colormap=:viridis)

    ax2d = Axis(fig[1, 2], xlabel="x", ylabel="y", title="Contour view", aspect=DataAspect())
    contourf!(ax2d, xs, ys, z; colormap=:plasma, levels=18)

    axforce = Axis(fig[1, 3], xlabel="x", ylabel="y", title="Force field", aspect=DataAspect())
    contourf!(axforce, xs, ys, z; colormap=:plasma, levels=18)

    # Coarser grid for arrows so the force field stays readable.
    arrow_xs = range(-3, 3; length=18)
    arrow_ys = range(-3, 3; length=18)
    positions = Point2f[]
    directions = Vec2f[]
    for y in arrow_ys, x in arrow_xs
        _, f = energy_and_force([x, y])
        push!(positions, Point2f(x, y))
        push!(directions, Vec2f(f[1], f[2]))
    end

    arrows!(axforce, positions, directions; arrowsize=10.0, lengthscale=0.05, color=:white)

    axlines = Axis(fig[1, 4], xlabel="x", ylabel="p", title="cv potential")
    lines!(axlines, xs, vec(sum(z; dims=2)); color=:blue)


    fig[0, :] = Label(fig, "2D flat double-well potential with two minima", fontsize=20)

    display(GLMakie.Screen(), fig)
    return fig
end

mutable struct Particle2D
    position::SVector{2,Float64}
    velocity::SVector{2,Float64}
    mass::Float64
end

function velocity_verlet_step!(particle::Particle2D, force::SVector{2,Float64}, dt::Float64)
    acceleration = force ./ particle.mass
    particle.position += particle.velocity .* dt .+ 0.5 .* acceleration .* dt^2

    potential, new_force = energy_and_force(particle.position)
    new_acceleration = new_force ./ particle.mass
    particle.velocity += 0.5 .* (acceleration + new_acceleration) .* dt

    return potential, new_force
end

function biased_velocity_verlet_step!(particle::Particle2D, force::SVector{2,Float64}, dt::Float64, bias_potential::Function)
    acceleration = force ./ particle.mass
    particle.position += particle.velocity .* dt .+ 0.5 .* acceleration .* dt^2

    potential, new_force = energy_and_force(particle.position, bias_potential)
    new_acceleration = new_force ./ particle.mass
    particle.velocity += 0.5 .* (acceleration + new_acceleration) .* dt

    return potential, new_force
end

function run_dynamics(; steps=2_000, dt=0.01, mass=1.0, x0=SVector(-2.0, 1.5), v0=SVector(0.0, 0.0))
    particle = Particle2D(x0, v0, mass)
    potential, force = energy_and_force(particle.position)

    trajectory = SVector{2,Float64}[particle.position]
    potential_energy = Float64[potential]
    kinetic_energy = Float64[0.5*particle.mass*sum(abs2, particle.velocity)]
    forces = [force]

    for _ in 1:steps
        potential, force = velocity_verlet_step!(particle, force, dt)
        push!(trajectory, particle.position)
        push!(potential_energy, potential)
        push!(kinetic_energy, 0.5 * particle.mass * sum(abs2, particle.velocity))
        push!(forces, force)
    end

    total_energy = potential_energy .+ kinetic_energy
    return (trajectory=trajectory, potential_energy=potential_energy, kinetic_energy=kinetic_energy, total_energy=total_energy, dt=dt, forces=forces)
end

function plot_dynamics(result)
    xs, ys, z = make_grid(potential_energy; xlim=(-3, 3), ylim=(-3, 3))

    fig = Figure(resolution=(1200, 500))
    ax_traj = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Trajectory", aspect=DataAspect())
    cf = contourf!(ax_traj, xs, ys, z; colormap=:plasma, levels=range(0, 5, length=18))
    Colorbar(fig[1, 2], cf; label="Potential energy", width=15)

    traj = reduce(hcat, result.trajectory)
    # lines!(ax_traj, traj[1, :], traj[2, :]; color=:white, linewidth=2)
    scatter!(ax_traj, traj[1, :], traj[2, :]; color=:white, markersize=1.0, label="Trajectory")
    scatter!(ax_traj, [traj[1, 1]], [traj[2, 1]]; color=:green, markersize=10, label="Start")
    scatter!(ax_traj, [traj[1, end]], [traj[2, end]]; color=:red, markersize=10, label="End")
    axislegend(ax_traj, position=:lb)

    ax_energy = Axis(fig[1, 3], xlabel="Step", ylabel="Energy", title="Energy")
    steps = 0:length(result.potential_energy)-1
    lines!(ax_energy, steps, result.potential_energy; label="Potential", color=:orange)
    lines!(ax_energy, steps, result.kinetic_energy; label="Kinetic", color=:dodgerblue)
    lines!(ax_energy, steps, result.total_energy; label="Total", color=:purple)
    axislegend(ax_energy, position=:rt)

    ax_force = Axis(fig[1, 4], xlabel="Step", ylabel="Force magnitude", title="Force magnitude")
    force_magnitudes = map(f -> norm(f), result.forces)
    lines!(ax_force, steps, force_magnitudes; color=:red)

    fig[0, :] = Label(fig, "Velocity-Verlet dynamics on the 2D double-well", fontsize=20)
    display(GLMakie.Screen(), fig)
    return fig
end

result = run_dynamics(; x0=SVector(0.01, 0.0), v0=SVector(0.0, 0.0), steps=5000_000, dt=0.0001)
plot_dynamics(result)


result = run_dynamics(; x0=SVector(2.0, -1.0), v0=SVector(0.0, 0.0), steps=100000, dt=0.01)
plot_dynamics(result)


function gaussian(x, cx, width)
    return (1 / (2 * π * prod(width))) * exp.(-0.5 * sum(((x .- cx) ./ width) .^ 2))
end

# silverman formula for bandwidth
function silverman_bandwidth(data)
    d = size(data, 1)
    n = size(data, 2)
    stddev = std(data; dims=2)
    return stddev .* (n * (d + 2) / 4)^(-1 / (d + 4))
end

cv(x) = [x[1]]
bandwidth = silverman_bandwidth(reduce(hcat, cv.(result.trajectory)))

kde_energy(x) = -log(sum(map(p -> gaussian(x, p, bandwidth), cv.(result.trajectory))))


kde_energy([0.0, 0.0])
gaussian([0.0, 0.0], [0.0, 0.0], bandwidth)


plot_potential(make_grid(potential_energy; xlim=(-3, 3), ylim=(-3, 3))...)
plot_potential(make_grid(kde_energy; xlim=(-3, 3), ylim=(-3, 3))...)

testgaussian(x) = gaussian(x, [0.0], bandwidth)
testgaussian([0.1, 0.2])
ForwardDiff.gradient(testgaussian, [0.0])
ForwardDiff.gradient(x -> gaussian(cv(x), [0.0], bandwidth), [0.1, 0.2])

function biased_dynamics(;
    steps=2_000, dt=0.01, mass=1.0, x0=SVector(-2.0, 1.5), v0=SVector(0.0, 0.0))

    # Initialize particle position and collect positions over dt as trajectory
    particle = Particle2D(x0, v0, mass)
    trajectory = SVector{2,Float64}[particle.position]
    gaussians = Function[(s->0.1)]
    weights = [0.1]
    biases = Function[(s->0.1)]

    potential, force = energy_and_force(particle.position)

    trajectory = SVector{2,Float64}[particle.position]
    potential_energy = Float64[potential]
    kinetic_energy = Float64[0.5*particle.mass*sum(abs2, particle.velocity)]
    forces = [force]
    Vs = Float64[0.0]
    Ps = Float64[0.0]


    γ = 1.04
    barrier_cap = 2.0
    ϵ = exp(-β * barrier_cap / (1 - 1 / γ))
    @showprogress for step in 1:steps
        pos = particle.position

        if step % 100 == 1
            w_n = exp(β * biases[end](cv(pos)))
            push!(weights, w_n)
            g_n(s) = gaussian(s, cv(pos), bandwidth)
            push!(gaussians, g_n)
        end


        potential, force = biased_velocity_verlet_step!(particle, force, dt, biases[end])

        P_n(s) = begin
            (sum(weights .* map(g -> g(s), gaussians))) / (sum(weights))
            # @tullio ps[i] := weights[i] * gaussians[i](s)
            # return sum(ps) / sum(weights)
        end
        V_n(s) = (1 - 1 / γ) * log(P_n(s) + ϵ) / β
        push!(biases, V_n)

        # push!(Ps, (sum(map(g -> g(cv(particle.position)), gaussians))) / (length(gaussians)))
        # push!(Vs, biases[end](cv(particle.position)))

        push!(trajectory, particle.position)
        push!(potential_energy, potential)
        push!(kinetic_energy, 0.5 * particle.mass * sum(abs2, particle.velocity))
        push!(forces, force)
    end

    total_energy = potential_energy .+ kinetic_energy
    return (trajectory=trajectory, potential_energy=potential_energy, kinetic_energy=kinetic_energy, total_energy=total_energy, dt=dt, forces=forces, biases=biases,
        gaussians=gaussians, weights=weights, bandwidth=bandwidth, Vs=Vs, Ps=Ps)
end


result = biased_dynamics(; x0=SVector(2.0, -1.0), v0=SVector(0.0, 0.0), steps=200000, dt=0.01);
plot_dynamics(result)
plot_potential(make_grid(result.biases[end]; xlim=(-3, 3), ylim=(-3, 3))...)



kde_energy(cv.(result.trajectory))
kde_cv_energy(s) = -log(sum(map(p -> gaussian(s, p, bandwidth), cv.(result.trajectory))))

let
    f = Figure()
    ax = Axis(f[1, 1], xlabel="x", ylabel="p", title="cv potential")
    xs = range(-3, 3; length=500)
    lines!(ax, xs, map(x -> result.biases[end]([x]), xs); color=:blue)
    lines!(ax, xs, map(x -> kde_cv_energy([x]), xs); color=:red)
    display(GLMakie.Screen(), f)
end

let
    f = Figure()
    ax = Axis(f[1, 1], xlabel="Step", ylabel="Estimated probability", title="Estimated probability over time")
    steps = 0:length(result.Ps)-1
    lines!(ax, steps, result.Ps; color=:green)
    lines!(ax, steps, result.Vs; color=:blue)
    display(GLMakie.Screen(), f)
end

result[:biases]
bs = map(x -> begin
        f, p = x
        # println("position", p, " cv: ", cv(p))
        f(cv(p))
    end, zip(result[:biases], result[:trajectory]))

let
    f = Figure()

    ax = Axis(f[1, 1], xlabel="Step", ylabel="Bias potential", title="Bias potential over time")
    steps = 0:length(bs)-1
    lines!(ax, steps, bs; color=:red)
    lines!(ax, steps, result.Vs; color=:blue, linestyle=:dash)
    display(GLMakie.Screen(), f)
end

cv(Particle2D(SVector(1.0, 2.0), SVector(0.0, 0.0), 1.0).position)