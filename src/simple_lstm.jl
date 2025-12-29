using ComponentArrays
using Lux
using NNlib: logsoftmax
using Random
using Images
using Optimisers
using Zygote
using OneHotArrays
using Lux.Training: TrainState, single_train_step!, AutoZygote


    # model = Lux.Chain(
    #     x -> reshape(x, size(x, 1), size(x, 2), size(x, 3), :),
    #     Conv((3, 3), 3 => 16, relu; pad=1),
    #     FlattenLayer(),
    #     Dense(prod(dims[1:2]) * 16, 64, relu),
    #     x -> reshape(x, size(x, 1), :, B),
    #     Lux.Recurrence(LSTMCell(64 => 32)),
    # )


function build_simple_lstm(batch_size; nclasses::Integer=10, rng=Random.default_rng())
    dims = (262, 1400, 3)
    model = Lux.Chain(
        x -> reshape(x, size(x, 1), size(x, 2), size(x, 3), :),
        Conv((3, 3), 3 => 16, relu; pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 16 => 32, relu; pad=1),
        MaxPool((2, 2)),
        FlattenLayer(),
        Dense((dims[1]÷4) * (dims[2]÷4) * 32, 64, relu),
        x -> reshape(x, size(x, 1), :, batch_size),
        Lux.Recurrence(LSTMCell(64 => 32)),
        Dense(32, nclasses),
    )
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end


function predict_logits(model, ps, st, x)
    y, _ = Lux.apply(model, x, ps, st)
    return y
end

# Example usage:
batch_size = 5
rng = MersenneTwister(42)
model, ps, st = build_simple_lstm(batch_size; rng)
x = dataset_all_img[1][1]
x = stack([x, x]) # now size is (H, W, C, S)
x = stack([x, x, x, x, x])  # now size is (H, W, C, S, batch_size)
logits = predict_logits(model, ps, st, x)
probs = logsoftmax(logits; dims=1)     # convert logits to log-probabilities
class = argmax(probs; dims=1)  # Get the predicted class for each image


###
conv1: 
maxpool1: (262, (1400-2)/2+1, 16, 15) --> (262, 700, 16, 15)
conv2







function train_on_all_orders(; epochs::Integer=200, batchsize::Integer=8, lr=1f-3, rng=MersenneTwister(0))
    @assert !isempty(dataset_all_img) "dataset_all_img is empty"
    @assert length(dataset_all_img) == length(dataset_all_order) "inputs and labels differ in length"
    nclasses = maximum(dataset_all_order)
    model, ps, st = build_simple_lstm(batchsize; nclasses=nclasses, rng)
    tstate = Training.TrainState(model, ps, st, Optimisers.Adam(lr))

    bce = BinaryCrossEntropyLoss(; logits=true)
    function batch_loss(model, ps, st, (x, y))
        logits, st = Lux.apply(model, x, ps, st)
        loss = sum(bce(logits, onehotbatch(y, 1:nclasses))) / length(y)
        return loss, st, logits
    end

    
    max_seq_len = maximum(lastindex.(dataset_all_img))
    for epoch in 1:epochs
        for seq_len in 1:max_seq_len
            idxs = filter(x -> 
                lastindex(dataset_all_img[x]) == seq_len, collect(eachindex(dataset_all_img)))
            shuffle!(rng, idxs)
            epoch_loss = 0f0
            correct = 0
            seen = 0
            for batch in Iterators.partition(idxs, batchsize)
                if length(batch) < batchsize
                    continue
                end
                x = Float32.(cat([cat(dataset_all_img[i]...; dims=4) for i in batch]...; dims=5))
                y = [dataset_all_order[i] for i in batch]

                _, loss, _, tstate = single_train_step!(AutoZygote(), batch_loss, (x, y), tstate)

                epoch_loss += Float32(loss) * length(y)
                logits, _ = Lux.apply(model, x, tstate.parameters, tstate.states)
                preds = [argmax(@view logits[:, j]) for j in 1:size(logits, 2)]
                correct += sum(preds .== y)
                seen += length(y)
            end
            println("epoch $epoch | loss=$(epoch_loss / seen) | acc=$(correct / seen)")
        end
    end
    return tstate
end

tstate = train_on_all_orders()

[dataset_all_img[2], dataset_all_img[4]]
(H,W, C, S, B)

dataset_all_img[2] -> (H,W, C,S)
cat(dataset_all_img[2]...; dims=4) #(H,W, C,S)

cat([cat(i...; dims=4) for i in [dataset_all_img[2], dataset_all_img[4]]]...; dims=5)

size(dataset_all_img[2][1])


let
    idx = 44
    x = reshape(dataset_all_img[idx], size(dataset_all_img[1])..., 1)
    logits = predict_logits(tstate.model, tstate.parameters, tstate.states, x)
    probs = logsoftmax(logits; dims=1)
    class = argmax(probs; dims=1)  # Get the pre

    println("Predicted class for image: $class")
    dataset[idx][1][1]
end

# using Lux, Random

# rng = MersenneTwister(42)
# conv = Conv((3, 3), 3 => 16, relu; pad=1)
# ps, st = Lux.setup(rng, conv)

# test_img = rand(rng, Float32, 4, 4, 3, 1)  # H×W×C×N
# y, st2 = Lux.apply(conv, test_img, ps, st)
# size(y)



dataset[1][1][1]

function demo_lstmcell(; input_dim=4, hidden_dim=6, seq_len=3, batch=2, rng=MersenneTwister(0))
    lstm = Lux.Recurrence(LSTMCell(input_dim => hidden_dim))
    ps, st = Lux.setup(rng, lstm)

    # Input needs to be feature×time×batch
    x = rand(rng, Float32, input_dim, seq_len, batch)
    println("Input size: ", size(x))

    # y has shape hidden_dim×seq_len×batch; st carries the final cell/hidden states
    y, st = Lux.apply(lstm, x, ps, st)
    println("Output size: ", size(y))
    return y, st
end
demo_lstmcell()



rgb_to_tensor(x) = permutedims(channelview(x) ./ 255.0, (2, 3, 1))
dataset_all_img = [[rgb_to_tensor.(i[1][1:j]) for j in 1:lastindex(i[2])] for i in dataset] |> Iterators.flatten |> collect
dataset_all_order = [i[2] for i in dataset] |> Iterators.flatten |> collect

# Vector{Array{H, W, S, C}}
dataset_all_img[2]

lastindex.(dataset_all_img)

size(dataset_all_img[1][1])
let
    dims = size(dataset_all_img[1][1])
    B = 5  # batch size
    model = Lux.Chain(
        x -> reshape(x, size(x, 1), size(x, 2), size(x, 3), :),
        Conv((3, 3), 3 => 16, relu; pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 16 => 32, relu; pad=1),
        MaxPool((2, 2)),
        FlattenLayer(),
        # x -> reshape(x, size(x, 1), :, B),
        # Lux.Recurrence(LSTMCell(64 => 32)),
    )
    ps, st = Lux.setup(rng, model)
    
    x = dataset_all_img[2][1]
    x = stack([x, x, x, x])
    x = stack([x, x, x, x, x])

    # y, _ = Lux.apply(model, x, ps, st)
    # size(y)
end
262 * 1400 * 16
(H, W, C, B)
(H, W, C, S, B)

1. image1 --> (H, W, C, B)
2. image2 --> (H, W, C, B)

(H?, W?, C, S, B) --> LSTM


(H, W, C_in, S, B) -> (Embed, S, B)

using Lux, Random

# 1. Define the dimensions
W, H, C = 64, 64, 3
T, B = 10, 4  # 10 frames, batch size of 4

# 2. Create the Reshape-Conv-Reshape pipeline
model = Chain(
    # Merge Time and Batch: (W, H, C, T, B) -> (W, H, C, T*B)
    x -> reshape(x, size(x, 1), size(x, 2), size(x, 3), :),
    
    # Standard 2D Conv (treats T*B as a large batch)
    Conv((3, 3), C => 16, relu; pad=SamePad()),
    
    # Split Time and Batch back: (W', H', C', T*B) -> (W', H', C', T, B)
    x -> reshape(x, size(x, 1), size(x, 2), size(x, 3), T, B)
)

# Setup
rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

# Dummy data: (64, 64, 3, 10, 4)
x = randn(Float32, W, H, C, T, B)
y, st = model(x, ps, st)

println("Output shape: ", size(y)) # Expected: (64, 64, 16, 10, 4)