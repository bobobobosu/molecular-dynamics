# MC sampling of a Lennard-Jones particle in a 2D box
using LinearAlgebra
using Distributions
using GLMakie
GLMakie.activate!(inline=false)
# Generate initial configuration of 7 particles in a 2D box, forming a hexagon for MC sampling

function hexagonal_particle2D(; center=(0.0, 0.0), spacing=1.0)
    # Center is (x0, y0)
    x0, y0 = center
    # Place 6 around, 1 in center
    angles = [0, π / 3, 2π / 3, π, 4π / 3, 5π / 3]
    positions = [[x0, y0]]
    for θ in angles
        x = x0 + spacing * cos(θ)
        y = y0 + spacing * sin(θ)
        push!(positions, [x, y] .+ rand(MvNormal(zeros(2), fill(1.0, 2))))
    end
    return positions
end


function circular_particle2D(; center=(0.0, 0.0), n=7, radius=1.0)
    # Arrange n particles approximately evenly spaced on a circle (no central atom)
    x0, y0 = center
    angles = range(0, 2π, length=n+1)[1:end-1]  # n evenly spaced angles around the circle
    positions = []
    for θ in angles
        x = x0 + radius * cos(θ)
        y = y0 + radius * sin(θ)
        # Add noise if desired
        noisy_position = [x, y]
        push!(positions, noisy_position)
    end
    return positions
end



function lennard_jones(r; ϵ=1.0, σ=1.0)
    sr6 = (σ / r)^6
    return 4 * ϵ * (sr6^2 - sr6)
end
let
    f_lj = Figure()
    ax_lj = Axis(f_lj[1, 1], xlabel="r", ylabel="Lennard-Jones Potential", title="Lennard-Jones Potential vs r")
    rs = range(0.8, 3.0; length=400)
    potentials = [lennard_jones(r) for r in rs]
    lines!(ax_lj, rs, potentials, color=:blue, label="LJ Potential")
    axislegend(ax_lj)
    xlims!(ax_lj, 0.8, 3.0)
    ylims!(ax_lj, -1.5, 2)
    display(f_lj)
end

let
    f_lj = Figure()
    ax_lj = Axis(f_lj[1, 1], xlabel="r", ylabel="Lennard-Jones Potential", title="Lennard-Jones Potential vs r")
    rs = range(0.8, 3.0; length=400)
    potentials = [lennard_jones(r) for r in rs]
    lines!(ax_lj, rs, potentials, color=:blue, label="LJ Potential")
    axislegend(ax_lj)
    xlims!(ax_lj, 0.8, 3.0)
    ylims!(ax_lj, -1.5, 2)
    display(f_lj)
end

function total_energy(hex_config)
    energy = 0.0
    for i in 1:lastindex(hex_config)
        for j in 1:lastindex(hex_config)
            if i != j
                diff = hex_config[i] - hex_config[j]
                r = norm(diff)
                # Robustness: skip if r is very small or not finite
                if !isfinite(r) || r < 1e-5
                    continue
                end
                energy += lennard_jones(r)
            end
        end
    end
    energy = energy / 2

    return energy
end

function mc_lj_move!(hex_config, T)
    hex_config_new = copy(hex_config)
    particle_index = rand(1:lastindex(hex_config))
    x′ = rand(MvNormal(hex_config[particle_index], fill(0.01, 2)))
    hex_config_new[particle_index] = x′
    acceptance_ratio = min(1, exp(-(total_energy(hex_config_new) - total_energy(hex_config)) / T))
    if rand() < acceptance_ratio
        hex_config[particle_index] = x′
    end
    return hex_config
end

total_energy([[0.0, 0.0], [1.0, 0.0]])
let
    hex_config = circular_particle2D(; center=(0.0, 0.0), n=7, radius=0.1)

    # hex_config = [[0.0, 0.0], [1.0, 0.0]]

    positions = Observable(reduce(hcat, hex_config))
    energies = Observable([total_energy(hex_config)])
    f = Figure()
    ax = Axis(f[1, 1], xlabel="x", ylabel="y", aspect=DataAspect())
    scatter!(ax, positions, markersize=20, color=:orange)
    ax2 = Axis(f[1, 2], xlabel="Frame", ylabel="Energy (kJ/mol)", title="Energy", yscale=log10)
    
    lines!(ax2, energies, label="Energy")
    axislegend(ax2)
    display(f)

    T = 1.0
    for i in 1:100000000000
        hex_config = mc_lj_move!(hex_config, T)
        # if i % 1000 == 0
        #     println("T = $T")
        #     T = T * 0.99
        # end
        e = total_energy(hex_config)
        if e < minimum(energies[])
            positions[] = reduce(hcat, hex_config)
            println("e = $e")
            sleep(0.1)
        end
        push!(energies[], e)
        if length(energies[]) > 100000
            popfirst!(energies[])
        end
        notify(positions)
        notify(energies)
        autolimits!(ax)
        autolimits!(ax2)
        yield()
        # sleep(0.1)
    end
end


using Turing
sample


function lj_potential(r; ε=1.0, σ=1.0)
    sr6 = (σ / r)^6
    return 4 * ε * (sr6^2 - sr6)
end

function total_energy(positions::Vector{Float64}; ε=1.0, σ=1.0)
    n_atoms = length(positions) ÷ 2
    E = 0.0
    for i in 1:n_atoms-1
        ri = positions[2i-1:2i]
        for j in i+1:n_atoms
            rj = positions[2j-1:2j]
            r = norm(ri - rj)
            E += lj_potential(r; ε=ε, σ=σ)
        end
    end
    return E
end

@model function lj_cluster_model(n_atoms::Int)
    # Initialize positions as a triangle
    pos_init = Vector{Float64}(undef, 2n_atoms)
    if n_atoms == 3
        # Equilateral triangle with side length ~1.1
        pos_init[1] = 0.0         # x1
        pos_init[2] = 0.0         # y1
        pos_init[3] = 1.1         # x2
        pos_init[4] = 0.0         # y2
        pos_init[5] = 0.55        # x3
        pos_init[6] = 1.1 * sqrt(3) / 2  # y3
    else
        # For more atoms, default to placing in a line (or make more elaborate triangle packing)
        for i in 1:n_atoms
            pos_init[2i-1] = (i - 1) * 1.1  # x
            pos_init[2i] = 0.0            # y
        end
    end
    positions ~ MvNormal(pos_init, 3.0 * I(length(pos_init)))

    # Compute total potential energy
    E = total_energy(positions)

    # Use a Boltzmann-like likelihood (lower energy → higher probability)
    temperature = 1.0
    Turing.@addlogprob!(-E / temperature)

    return positions
end

n_atoms = 2
model = lj_cluster_model(n_atoms)
chain = sample(model, NUTS(), 10000)
summarystats(chain)
using DataFrames
pos = reshape(DataFrame(mean(chain))[!, :mean], 2, n_atoms)

let
    f = Figure()
    ax = Axis(f[1, 1], xlabel="x", ylabel="y", aspect=DataAspect())
    scatter!(ax, pos[1, :], pos[2, :], markersize=20, color=:orange)
    display(f)
end


let
    f = Figure()
    ax = Axis(f[1, 1], xlabel="x", ylabel="y")
    init_pos = reduce(vcat, hexagonal_particle2D(center=(0.0, 0.0), spacing=1.0))
    pos = reshape(init_pos, 2, 7)
    scatter!(ax, pos[1, :], pos[2, :], markersize=20, color=:orange)
    display(f)
end
