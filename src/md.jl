using Unitful
using StaticArrays
using ForwardDiff
using LinearAlgebra
using Tullio
using GLMakie
using CUDA, KernelAbstractions, Zygote
GLMakie.activate!()

mutable struct Particle
    position::SVector{3,typeof(1.0u"nm")}
    velocity::SVector{3,typeof(1.0u"nm/ps")}
    mass::typeof(1.0u"u")
end

begin
    # Periodic boundary conditions - box dimensions
    BOX_SIZE = 10u"nm"
    MASS_ARGON = 39.948u"u"
    function offset_bc(posA::AbstractVector, posB::AbstractVector)
        dr = posA - posB
        dr = dr .- BOX_SIZE .* round.(dr ./ BOX_SIZE)
        return dr
    end
    offset(particleA::Particle, particleB::Particle) = offset_bc(particleA.position, particleB.position)

    # Create a 3D grid with spacing of 1 angstrom (1u"Å")
    function create_3d_grid(xmin, xmax, ymin, ymax, zmin, zmax)
        xs = xmin:10u"Å":xmax-1.0u"Å"
        ys = ymin:10u"Å":ymax-1.0u"Å"
        zs = zmin:10u"Å":zmax-1.0u"Å"
        grid = [(x, y, z) for x in xs, y in ys, z in zs]
        return grid
    end

    md_grid = create_3d_grid(0u"Å", BOX_SIZE, 0u"Å", BOX_SIZE, 0u"Å", BOX_SIZE)


    using Random

    # Create particles with random positions in each grid cell
    temperature = 298.0u"K"
    boltzmann_constant = 1.380649e-23u"J/K"
    particles = Particle[]
    for grid_point in vec(md_grid)
        position = grid_point
        scaling_factor = sqrt((boltzmann_constant * temperature) / (MASS_ARGON))
        velocity = (
            (rand() - 0.5) * scaling_factor,
            (rand() - 0.5) * scaling_factor,
            (rand() - 0.5) * scaling_factor
        )
        # velocity = (0.0u"nm/ps", 0.0u"nm/ps", 0.0u"nm/ps")
        mass = MASS_ARGON

        push!(particles, Particle(position, velocity, mass))
    end
    BOX_SIZE_cu = repeat([BOX_SIZE], length(particles))

    # Remove the mean momentum from the particles
    begin
        # Compute the mean momentum vector (unit: momentum)
        total_momentum = zero(particles[1].mass .* particles[1].velocity)
        for particle in particles
            total_momentum += particle.mass .* particle.velocity
        end
        mean_momentum = total_momentum ./ length(particles) # unit: momentum

        # Subtract mean momentum from each particle (adjust velocity)
        for particle in particles
            particle.velocity -= mean_momentum ./ particle.mass
        end
    end

    """
        lennard_jones(r; ϵ=1.0, σ=1.0)

    Compute the Lennard-Jones potential energy for a given distance `r`.

    # Arguments
    - `r`: Distance between two particles (in the same units as σ).
    - `ϵ`: Depth of the potential well (default 1.0).
    - `σ`: Finite distance at which the inter-particle potential is zero (default 1.0).

    # Returns
    - Lennard-Jones potential energy at distance `r`.
    """
    function lennard_jones(r; ϵ=1.0u"kJ/mol", σ=1.0u"nm")
        sr6 = (σ / r)^6
        return 4 * ϵ * (sr6^2 - sr6)
    end

    function lennard_jones(displacement::SVector{3,typeof(1.0u"nm")}; ϵ=0.997u"kJ/mol", σ=0.34u"nm")
        r = sqrt(sum(abs2, displacement))
        sr6 = (σ / r)^6
        return 4 * ϵ * (sr6^2 - sr6)
    end
    function lennard_jones(displacement::AbstractVector; ϵ=0.997, σ=0.34)
        r = sqrt(sum(abs2, displacement))
        sr6 = (σ / r)^6
        return 4 * ϵ * (sr6^2 - sr6)
    end
    function lennard_jones_force(particleA::Particle, particleB::Particle)
        displacement = offset(particleA, particleB)
        -ForwardDiff.gradient(lennard_jones, ustrip.(u"nm", displacement)) .* u"kJ/(mol*nm)"
    end

    @tullio forces[i] := begin
        if i != j
            lennard_jones_force(particles[i], particles[j])
        else
            zero(typeof(lennard_jones_force(particles[1], particles[2])))
        end
    end

    function verlet_step!(particles, forces, dt)
        particle_positions = ((x -> x.position).(particles))

        # Velocity-Verlet algorithm
        # Step 1: Update positions using current velocities and forces
        for i in eachindex(particles)
            acceleration = (forces[i] / particles[i].mass) |> collect
            acceleration = ustrip.(u"kJ *nm^-1 *mol^-1 *u^-1", acceleration) * u"nm/ps^2"
            particles[i].position += particles[i].velocity .* dt .+ 0.5 .* acceleration .* dt .^ 2

            # Apply periodic boundary conditions
            particles[i].position = mod.(particles[i].position, BOX_SIZE)
        end

        # Step 2: Calculate new forces at updated positions
        @tullio new_forces[i] := begin
            dr = particle_positions[i] - particle_positions[j]
            dr = dr .- BOX_SIZE_cu[i] .* round.(dr ./ BOX_SIZE_cu[i])
            r = -ForwardDiff.gradient(lennard_jones, ustrip.(u"nm", dr)) .* u"kJ/(mol*nm)"
            i != j ? r : zero(r)
        end
        new_forces = new_forces |> collect

        # Step 3: Update velocities using average of old and new forces
        for i in eachindex(particles)
            old_acceleration = forces[i] / particles[i].mass
            new_acceleration = new_forces[i] / particles[i].mass
            average_acceleration = (old_acceleration + new_acceleration) / 2
            average_acceleration = ustrip.(u"kJ *nm^-1 *mol^-1 *u^-1", average_acceleration) * u"nm/ps^2"
            particles[i].velocity += average_acceleration .* dt
        end
        forces .= new_forces


        @tullio kinetic_energy := 0.5 * particles[i].mass * sum(abs2, particles[i].velocity) # in nm^2 u ps^-2
        kinetic_energy = kinetic_energy.val * 1.66053906660e-21 # in kJ/mol
        @tullio potential_energy[i] := begin
            dr = particle_positions[i] - particle_positions[j]
            dr = dr .- BOX_SIZE_cu[i] .* round.(dr ./ BOX_SIZE_cu[i])
            r = lennard_jones(ustrip.(u"nm", dr)) * u"kJ/mol"
            i != j ? r : zero(r)
        end
        potential_energy = potential_energy |> collect |> sum
        Na = 6.02214076e23  # Avogadro's number (mol⁻¹)
        potential_energy = potential_energy.val * 1000 / Na
        return potential_energy, kinetic_energy
    end

    let
        fig = Figure()
        ax = Axis3(fig[1, 1], xlabel="x (Å)", ylabel="y (Å)", zlabel="z (Å)")

        # Extract grid points and convert to Ångström
        xs = unique([uconvert(u"Å", x[1]).val for x in vec(md_grid)])
        ys = unique([uconvert(u"Å", x[2]).val for x in vec(md_grid)])
        zs = unique([uconvert(u"Å", x[3]).val for x in vec(md_grid)])

        # Draw lines along x for each (y, z)
        for y in ys, z in zs
            lines!(ax, xs, fill(y, length(xs)), fill(z, length(xs)), color=:gray, linewidth=1)
        end

        # Draw lines along y for each (x, z)
        for x in xs, z in zs
            lines!(ax, fill(x, length(ys)), ys, fill(z, length(ys)), color=:gray, linewidth=1)
        end

        # Draw lines along z for each (x, y)
        for x in xs, y in ys
            lines!(ax, fill(x, length(zs)), fill(y, length(zs)), zs, color=:gray, linewidth=1)
        end

        # Draw particles
        positions = Observable(reduce(hcat, [[uconvert(u"Å", x).val for x in particle.position] for particle in particles]))
        scatter!(ax, positions, markersize=10, color=:blue)

        # # Plot force vectors as arrows for every 5th particle
        # for i in 1:5:length(particles)
        #     pos = [uconvert(u"Å", x).val for x in particles[i].position]
        #     force = [ustrip(x) for x in forces[i]]
        #     # Normalize force for visualization, scale for arrow length
        #     norm_force = norm(force)
        #     if norm_force > 0
        #     arrow_dir = force / norm_force
        #     arrow_length = 50  # adjust for visual clarity
        #     arrow_vec = arrow_dir * arrow_length
        #     arrows2d!(ax, [pos[1]], [pos[2]], [pos[3]],
        #         [arrow_vec[1]], [arrow_vec[2]], [arrow_vec[3]], color=:red)
        #     end
        # end

        display(GLMakie.Screen(), fig)


        kinetic_energy = Observable(Float64[])
        potential_energy = Observable(Float64[])
        total_energy = Observable(Float64[])

        ax2 = Axis(fig[1, 2], xlabel="Frame", ylabel="Energy (kJ/mol)", title="Energy")
        lines!(ax2, kinetic_energy, label="Kinetic Energy")
        lines!(ax2, potential_energy, label="Potential Energy")
        lines!(ax2, total_energy, label="Total Energy")
        axislegend(ax2)

        global traj
        traj = []
        prev_ts = time()
        while true
            
            if time() - prev_ts > 1.0
                positions[] .= reduce(hcat, [[uconvert(u"Å", x).val for x in particle.position] for particle in particles])
                notify(positions)
                prev_ts = time()
            end
            # sleep(0.001)
            dt = 1.0u"fs"
            pn_J, kn_J = verlet_step!(particles, forces, dt)

            elapsed_hours = (time() - prev_ts) / 3600
            sim_per_hr_ps = elapsed_hours > 0 ? (length(traj) / 1000) / elapsed_hours : 0.0
            if sim_per_hr_ps > 0
                @info "Simulated $(sim_per_hr_ps) ps/hr"
            end


            push!(kinetic_energy[], kn_J)
            push!(potential_energy[], pn_J)
            push!(total_energy[], kinetic_energy[][end] + potential_energy[][end])
            notify(kinetic_energy)
            notify(potential_energy)
            notify(total_energy)
            autolimits!(ax2)
            push!(traj, positions[] |> collect)
        end
    end
end

using Serialization
serialize("traj_gpu.jls", Dict("traj" => traj))