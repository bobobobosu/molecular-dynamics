# MC sampling of a 2D Ising model
using LinearAlgebra
using Distributions
using Observables
using GLMakie
GLMakie.activate!(inline=false)

function neighbor_indices(i, j, x_max, y_max)
    up = mod1(i - 1, x_max)
    down = mod1(i + 1, x_max)
    left = mod1(j - 1, y_max)
    right = mod1(j + 1, y_max)
    return [(up, j), (down, j), (i, left), (i, right)]
end

function compute_energy(spins, J, h)
    x_max, y_max = size(spins)
    energy = 0.0
    for (i, j) in Iterators.product(1:x_max, 1:y_max)
        s = spins[i, j]
        for n in neighbor_indices(i, j, x_max, y_max)
            energy -= 0.5 * J * s * spins[n...]  # each pair counted twice
        end
        energy -= h * s
    end
    return energy
end

function MC_move!(spins, T, J, h)
    x_max, y_max = size(spins)
    i = rand(1:x_max)
    j = rand(1:y_max)
    E_before = compute_energy(spins, J, h)
    spins[i, j] = -spins[i, j]  # Propose flip
    E_after = compute_energy(spins, J, h)
    ΔE = E_after - E_before
    if ΔE <= 0 || rand() < exp(-ΔE / T)
        return true
    else
        spins[i, j] = -spins[i, j]  # Revert flip
        return false
    end
end



begin
    # 1. Generate initial configuration of spins in a 2D lattice
    # On a 50x50 grid, randomly assign +1 or -1 to each site
    x, y = 20, 20
    T, J, h = 2.5, 1.0, 0.0

    spins_obs = Observable(2 .* rand(Bernoulli(0.5), x, y) .- 1)
    total_energy = Observable([compute_energy(spins_obs[], J, h)])


    fig = Figure()

    ax = Axis(fig[1, 1], title="2D Ising Model Spins", xlabel="X", ylabel="Y")
    heatmap!(ax, spins_obs; colormap=:coolwarm, colorrange=(-1, 1))

    ax_energy = Axis(fig[1, 2], title="Total Energy", xlabel="MC Steps", ylabel="Energy")
    lines!(ax_energy, total_energy, color=:blue)

    ax_avg_spin = Axis(fig[1, 3], title="Average Spin", xlabel="MC Steps", ylabel="⟨σ⟩")
    avg_spin = Observable([mean(spins_obs[])])
    lines!(ax_avg_spin, avg_spin, color=:red)

    ax_acceptrate = Axis(fig[2, 1], title="Acceptance Rate", xlabel="MC Steps", ylabel="Rate")
    accept_rate = Observable([0.0])
    lines!(ax_acceptrate, accept_rate, color=:green)


    T_slider = Slider(fig[3, 1:3], range=0.1:0.1:1000.0, startvalue=0.00001)
    J_slider = Slider(fig[4, 1:3], range=0.0:0.1:10000.0, startvalue=J)
    h_slider = Slider(fig[5, 1:3], range=-2.0:0.1:2.0, startvalue=0.0)
    display(fig)

    step = 0
    acceptence = []
    while true
        step += 1
        T = T_slider.value[]
        J = J_slider.value[]
        h = h_slider.value[]

        accept = MC_move!(spins_obs[], T, J, h)
        push!(acceptence, accept ? 1.0 : 0.0)
        push!(total_energy[], compute_energy(spins_obs[], J, h))
        push!(avg_spin[], mean(spins_obs[]))
        push!(accept_rate[], mean(acceptence[end - min(step, 100) + 1:end]))

        if step % 500 == 0
            notify(avg_spin)
            notify(spins_obs)
            notify(total_energy)
            notify(accept_rate)
            autolimits!(ax_energy)
            autolimits!(ax_avg_spin)
            autolimits!(ax_acceptrate)
        end

        if step > 100_000
            popat!(total_energy[], 1)
            popat!(avg_spin[], 1)
            popat!(accept_rate[], 1)
            popat!(acceptence, 1)
        end
        yield()
    end
end