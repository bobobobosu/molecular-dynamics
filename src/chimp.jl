using Random
using GLMakie
using Colors

GLMakie.activate!(inline=false)

const N_SLOTS = 10

"""
    chimp_test_1d(; numbers=8)

Launch a simple 1D chimp memory test: numbers are dealt randomly onto 20
buttons in a row. After the first correct tap, the labels disappear and you
have to finish in ascending order from memory.
"""
function chimp_test_1d(; numbers=8)
    numbers = clamp(numbers, 2, N_SLOTS)

    # Observables driving the UI
    labels = Observable(fill("", N_SLOTS))           # current labels per slot
    colors = Observable(fill(:gray92, N_SLOTS))      # button colors per slot
    colors_reset = copy(colors[])                     # default colors
    order_obs = Observable(Int[])                    # positions to hit in order
    target = Observable(1)                           # 1-based index into `order_obs`
    show_numbers = Observable(true)                  # whether labels are visible
    status = Observable("Press Start to deal numbers.")

    f = Figure(; fontsize=18, size=(800, 150))
    control = f[1, 1] = GridLayout(tellwidth=true)
    row = f[2, 1] = GridLayout()

    Label(control[1, 1], "Numbers this round")
    slider = Slider(control[1, 2], range=2:1:N_SLOTS, width=300, value=numbers)
    Label(control[1, 3], lift(x -> "Count: $(round(Int, x))", slider.value))
    start_btn = Button(control[1, 4], label="Start / reshuffle", width=160)
    status_label = Label(control[2, 1:4], status, halign=:left)

    function new_round!(n::Int)
        n = clamp(n, 2, N_SLOTS)
        order = randperm(N_SLOTS)[1:n]
        dealt_labels = fill("", N_SLOTS)
        for (i, pos) in enumerate(order)
            dealt_labels[pos] = string(i)
        end
        dealt_colors = fill(:gray92, N_SLOTS)
        for pos in order
            dealt_colors[pos] = :lightblue
        end
        labels[] = dealt_labels
        colors[] = dealt_colors
        colors_reset .= dealt_colors
        order_obs[] = order
        target[] = 1
        show_numbers[] = true
        status[] = "Memorize: first tap hides numbers. Need $n taps."
        return RGB.(GLMakie.colorbuffer(f.scene)), order
    end

    on(start_btn.clicks) do _
        new_round!(round(Int, slider.value[]))
    end

    display_labels = lift(show_numbers, labels) do show, labs
        show ? labs : fill(" ", length(labs))
    end

    actions = []
    for i in 1:N_SLOTS
        label_obs = lift(display_labels) do labs
            labs[i]
        end
        color_obs = lift(colors) do cs
            cs[i]
        end
        btn = Button(row[1, i], label=label_obs, width=52, height=52, buttoncolor=color_obs)
        press_button() = begin
            if isempty(order_obs[])
                status[] = "Press Start to deal numbers."
                return RGB.(GLMakie.colorbuffer(f.scene))
            end
            # Hide labels on the first user click of any button
            if show_numbers[]
                show_numbers[] = false
            end
            slot_label = labels[][i]
            if isempty(slot_label)
                status[] = "No number here. Try another slot."
                return RGB.(GLMakie.colorbuffer(f.scene))
            end
            expected_pos = order_obs[][target[]]
            if i != expected_pos
                show_numbers[] = true
                status[] = "Wrong slot! Press Start to try a new round."
                target[] = 1
                colors[] .= colors_reset
                notify(colors)
                return RGB.(GLMakie.colorbuffer(f.scene))
            end
            # Correct slot: mark green and keep going
            cs = copy(colors[])
            cs[i] = :gray92
            colors[] = cs
            if target[] == 1
                show_numbers[] = false
            end
            if target[] == length(order_obs[])
                show_numbers[] = false
                status[] = "Perfect run! Press Start to shuffle again."
                target[] = 1
                return RGB.(GLMakie.colorbuffer(f.scene))
            end
            target[] += 1
            status[] = "Good. Next number: $(target[])."
            return RGB.(GLMakie.colorbuffer(f.scene))
        end
        on(btn.clicks) do _
            press_button()
        end
        push!(actions, press_button)
    end

    function reward()
        -1 * count(c -> c == :lightblue, colors[])
    end

    display(f)
    return f, new_round!, actions, reward
end


# Gym-style RL wrapper
mutable struct ChimpEnv
    fig; actions; reset_fn; reward_fn
    order::Vector{Int}; target::Int; done::Bool; n::Int
end

ChimpEnv(; n=5) = begin
    f, r!, a, rw = chimp_test_1d(numbers=n)
    ChimpEnv(f, a, r!, rw, Int[], 1, true, n)
end

action_space(::ChimpEnv) = 1:N_SLOTS
render(env::ChimpEnv) = RGB.(GLMakie.colorbuffer(env.fig.scene))
is_done(env::ChimpEnv) = env.done
sample_action(::ChimpEnv) = rand(1:N_SLOTS)
optimal_action(env::ChimpEnv) = env.done ? 1 : env.order[env.target]

function reset!(env::ChimpEnv; n=env.n)
    env.n = clamp(n, 2, N_SLOTS)
    obs, env.order = env.reset_fn(env.n)
    env.target, env.done = 1, false
    obs, (order=env.order,)
end

function step!(env::ChimpEnv, action::Int)
    env.done && error("Call reset! first")
    obs = env.actions[clamp(action, 1, N_SLOTS)]()
    correct = (action == env.order[env.target])
    if correct
        done = env.target == length(env.order)
        reward = done ? 110.0 : 10.0
        env.target = done ? 1 : env.target + 1
        env.done = done
    else
        reward, env.done = -10.0, true
    end
    obs, reward, env.done, false, (action=action, correct=correct)
end

# Demo
function demo()
    env = ChimpEnv(n=4)
    
    # Optimal policy run
    obs, info = reset!(env)
    println("Order to memorize: ", info.order)
    total = 0.0
    while !is_done(env)
        a = optimal_action(env)
        obs, r, done, _, info = step!(env, a)
        total += r
        println("Action: $a, Reward: $r, Correct: $(info.correct)")
    end
    println("Total reward (optimal): $total\n")
    
    # # Random policy run
    # reset!(env)
    # total = 0.0
    # steps = 0
    # while !is_done(env) && steps < 50
    #     a = sample_action(env)
    #     _, r, _, _, _ = step!(env, a)
    #     total += r
    #     steps += 1
    # end
    # println("Total reward (random, $steps steps): $total")
    render(env)
end
demo()



test_env= ChimpEnv(n=5)
obs, info = reset!(test_env)

render(test_env)
step!(test_env, 5)
render(test_env)
is_done(test_env)