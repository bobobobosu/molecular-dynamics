using DifferentialEquations
using GLMakie


function lorenz!(du,u,p,t)
 du[1] = 10.0(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
end
u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
prob = ODEProblem(lorenz!,u0,tspan)

# Test that it worked
using OrdinaryDiffEq
sol = solve(prob,Tsit5(); saveat=0.01)

GLMakie.activate!(inline=false)

sol_array = Array(sol)
t = sol.t
x = sol_array[1, :]
y = sol_array[2, :]
z = sol_array[3, :]

let
    fig = Figure()
    ax1 = Axis(fig[1, 1], xlabel = "t", ylabel = "x(t)", title = "Lorenz x-component")
    lines!(ax1, t, x, color = :red)

    ax2 = Axis(fig[1, 2], xlabel = "t", ylabel = "y(t)", title = "Lorenz y-component")
    lines!(ax2, t, y, color = :blue)

    ax3 = Axis(fig[2, 1], xlabel = "t", ylabel = "z(t)", title = "Lorenz z-component")
    lines!(ax3, t, z, color = :green)

    ax4 = Axis3(fig[2, 2], xlabel = "x", ylabel = "y", zlabel = "z", title = "Lorenz Attractor")
    lines!(ax4, x, y, z, color = :purple)

    fig[0, :] = Label(fig, "Lorenz System Integration", fontsize = 22)
    display(fig)
end
