using ComponentArrays
using Lux
using NNlib: logsoftmax
using Random
using Images
using Optimisers
using Zygote
using OneHotArrays
using Lux.Training: TrainState, single_train_step!, AutoZygote

"""
    build_simple_cnn(; nclasses=10, rng=Random.default_rng())

Return a tiny convolutional classifier for fixed-size RGB inputs of shape
`31×350×3×batch`, producing logits for `nclasses` classes. The architecture is
intentionally small: two conv+pool blocks, a flatten, and two dense layers.
"""
function build_simple_cnn(; nclasses::Integer=10, rng=Random.default_rng())
    model = Lux.Chain(
        Conv((3, 3), 3 => 16, relu; pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 16 => 32, relu; pad=1),
        MaxPool((2, 2)),
        FlattenLayer(),
        # With input 31×350, after two 2×2 pools we have 7×87 spatial dims
        Dense(7 * 87 * 32, 64, relu),
        Dense(64, nclasses),              # logits; apply softmax externally if needed
    )
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end

"""
    predict_logits(model, ps, st, x)

Run a forward pass on a batch of RGB images `x` (31×350×3×batch) and return the
raw logits. Apply `logsoftmax` or `softmax` downstream to obtain probabilities.
"""
function predict_logits(model, ps, st, x)
    y, _ = Lux.apply(model, x, ps, st)
    return y
end

dataset_first_img = [permutedims(channelview(i[1][1][140:4:end, 1:4:end]) ./ 255.0, (2, 3, 1)) for i in dataset]
dataset_first_order = [i[2][1] for i in dataset]

dataset_first_img[1]

# Example usage:
rng = MersenneTwister(42)
model, ps, st = build_simple_cnn(; rng)
# x = rand(rng, Float32, 31, 350, 3, 8)  # 8 RGB images at 31×350
x = reshape(dataset_first_img[1], size(dataset_first_img[1])..., 1)
logits = predict_logits(model, ps, st, x)
probs = logsoftmax(logits; dims=1)     # convert logits to log-probabilities
class = argmax(probs; dims=1)  # Get the predicted class for each image

function train_on_first_orders(; epochs::Integer=200, batchsize::Integer=8, lr=1f-3, rng=MersenneTwister(0))
    @assert !isempty(dataset_first_img) "dataset_first_img is empty"
    @assert length(dataset_first_img) == length(dataset_first_order) "inputs and labels differ in length"
    nclasses = maximum(dataset_first_order)
    model, ps, st = build_simple_cnn(; nclasses=nclasses, rng)
    tstate = Training.TrainState(model, ps, st, Optimisers.Adam(lr))

    bce = BinaryCrossEntropyLoss(; logits=true)
    function batch_loss(model, ps, st, (x, y))
        logits, st = Lux.apply(model, x, ps, st)
        loss = sum(bce(logits, onehotbatch(y, 1:nclasses))) / length(y)
        return loss, st, logits
    end

    idxs = collect(eachindex(dataset_first_img))
    for epoch in 1:epochs
        shuffle!(rng, idxs)
        epoch_loss = 0f0
        correct = 0
        seen = 0
        for batch in Iterators.partition(idxs, batchsize)
            x = cat([Float32.(dataset_first_img[i]) for i in batch]...; dims=4)
            y = [dataset_first_order[i] for i in batch]

            _, loss, _, tstate = single_train_step!(AutoZygote(), batch_loss, (x, y), tstate)

            epoch_loss += Float32(loss) * length(y)
            logits, _ = Lux.apply(model, x, tstate.parameters, tstate.states)
            preds = [argmax(@view logits[:, j]) for j in 1:size(logits, 2)]
            correct += sum(preds .== y)
            seen += length(y)
        end
        println("epoch $epoch | loss=$(epoch_loss / seen) | acc=$(correct / seen)")
    end
    return tstate
end

tstate = train_on_first_orders()


let
    idx = 44
    x = reshape(dataset_first_img[idx], size(dataset_first_img[1])..., 1)
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
