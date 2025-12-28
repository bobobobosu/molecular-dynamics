using Random
using GLMakie

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
    order_obs = Observable(Int[])                    # positions to hit in order
    target = Observable(1)                           # 1-based index into `order_obs`
    show_numbers = Observable(true)                  # whether labels are visible
    status = Observable("Press Start to deal numbers.")

    f = Figure(resolution=(1200, 320), fontsize=16)
    control = f[1, 1] = GridLayout(tellwidth=true)
    row = f[2, 1] = GridLayout()

    Label(control[1, 1], "Numbers this round")
    slider = Slider(control[1, 2], range=2:1:N_SLOTS, width=300)
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
        order_obs[] = order
        target[] = 1
        show_numbers[] = true
        status[] = "Memorize: first tap hides numbers. Need $n taps."
    end

    on(start_btn.clicks) do _
        new_round!(round(Int, slider.value[]))
    end

    display_labels = lift(show_numbers, labels) do show, labs
        show ? labs : fill(" ", length(labs))
    end

    for i in 1:N_SLOTS
        label_obs = lift(display_labels) do labs
            labs[i]
        end
        color_obs = lift(colors) do cs
            cs[i]
        end
        btn = Button(row[1, i], label=label_obs, width=52, height=52, buttoncolor=color_obs)
        on(btn.clicks) do _
            if isempty(order_obs[])
                status[] = "Press Start to deal numbers."
                return
            end
            slot_label = labels[][i]
            if isempty(slot_label)
                status[] = "No number here. Try another slot."
                return
            end
            expected_pos = order_obs[][target[]]
            if i != expected_pos
                show_numbers[] = true
                status[] = "Wrong slot! Press Start to try a new round."
                target[] = 1
                return
            end
            if target[] == 1
                show_numbers[] = false
            end
            if target[] == length(order_obs[])
                show_numbers[] = true
                status[] = "Perfect run! Press Start to shuffle again."
                target[] = 1
                return
            end
            target[] += 1
            status[] = "Good. Next number: $(target[])."
        end
    end

    display(f)
    return f
end
chimp_test_1d()
