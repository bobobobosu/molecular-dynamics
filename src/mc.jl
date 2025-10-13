using Distributions
using GLMakie
GLMakie.activate!(inline=false)

dist = Normal(10.0, 10.0)

N_samples = 100000
samples = []

x_t = 1.0
for i in 1:N_samples
    proposal_dist = Normal(x_t, 1.0)
    x_t′ = rand(proposal_dist)

    acceptance_ratio = min(1, 
    exp(logpdf(dist, x_t′) - logpdf(dist, x_t))
)

    if rand() < acceptance_ratio
        x_t = x_t′
    end

    push!(samples, x_t)
end


let
    f = Figure()
    ax = Axis(f[1, 1])
    hist1 = hist!(ax, rand(dist, N_samples), bins=1000, label="True Distribution")
    hist2 = hist!(ax, samples, bins=1000, label="MCMC Samples")
    xlims!(ax, -100, 100)
    axislegend(ax)
    display(f)
end

# Generate initial configuration of 7 particles in a 2D box, forming a hexagon for MC sampling

function hexagonal_particles2D(center=(0.0,0.0), spacing=1.0)
    # Center is (x0, y0)
    x0, y0 = center
    # Place 6 around, 1 in center
    angles = [0, π/3, 2π/3, π, 4π/3, 5π/3]
    positions = [[x0, y0]]
    for θ in angles
        x = x0 + spacing * cos(θ)
        y = y0 + spacing * sin(θ)
        push!(positions, [x, y])
    end
        return positions
    end

# Example usage:
hex_config = hexagonal_particles2D(center=(10.0, 10.0), spacing=1.0)

# Optionally visualize
let
    f = Figure()
    ax = Axis(f[1,1], xlabel="x", ylabel="y", aspect=DataAspect())
    xs = [p[1] for p in hex_config]
    ys = [p[2] for p in hex_config]
    scatter!(ax, xs, ys, markersize=20, color=:orange, label="Hexagonal configuration")
    xlims!(ax, 7, 13)
    ylims!(ax, 7, 13)
    axislegend(ax)
    display(f)
end
