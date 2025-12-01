using Distributions
using ForwardDiff
using LinearAlgebra

q = Normal(0, 1)

x = rand(q)

pdf(q, x)

a = 2.0
b = 4.0
ϕ(x) = a .* x .+ b
ϕ⁻¹(y) = (y .- b) ./ a
randp() = ϕ(rand(q))



qy(y) = pdf(q, ϕ⁻¹(y)) * det(ForwardDiff.derivative(ϕ⁻¹, y))
qy(0.0)

using GLMakie
GLMakie.activate!(inline=false)


let 
    f = Figure()
    ax = Axis(f[1, 1])
    hist!(ax, [rand(q) for _ in 1:10000]; bins=100, color=:red)
    hist!(ax, [randp() for _ in 1:10000]; bins=100, color=:blue)
    xlims!(ax, -20, 20)
    ax = Axis(f[2, 1])
    lines!(ax, -20:0.1:20, x -> pdf(q, x), color=:red, linewidth=3)
    lines!(ax, -20:0.1:20, y -> qy(y), color=:blue, linewidth=3)
    xlims!(ax, -20, 20)
    display(f)
end



