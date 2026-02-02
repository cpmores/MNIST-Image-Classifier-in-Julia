module Model

using ..Layer
export Chain, Mod 
export forward, backward
export train, test
export decode_y_true

function encode_y_true(y_true::UInt8, num)
    ret = zeros(Float32, num)
    index::UInt8 = 1
    if y_true <= num - 1 && y_true >= 0 
        index = y_true + 1
    end

    ret[index] = 1.0f0
    return ret
end

function decode_y_true(ys::Array{Float32})
    index::UInt8 = 0
    max = 0.0f0
    for (i, y) in enumerate(ys)

        if y > max
            max = y
            index = i - 1
        end
    end

    return index
end

struct Chain
    lossLayer::SoftmaxWithCrossEntropyLoss
    layers::Vector{AbstractLayer}
    cacheList::Vector{LayerCache}
    step::Float32

    function Chain(layers...; step = 0.1f0, lossLayer = SoftmaxWithCrossEntropyLoss())
        layer_vec = AbstractLayer[]
        cache_vec = LayerCache[]
        for layer in layers

            if layer isa AbstractLayer
                
                push!(layer_vec, layer)
                push!(cache_vec, LayerCache())
            end
        end

        push!(cache_vec, LayerCache())
        new(lossLayer, layer_vec, cache_vec, step)
    end 
end

struct Mod
    chain::Chain

    function Mod(chain::Chain)
        new(chain)
    end
end

# needs implement
function train end
function test end

function forward(chain::Chain, image, label)
    input = copy(image)
    y_true = encode_y_true(label, 10)
    for (index, layer) in enumerate(chain.layers)

        output, cache = Layer.forward(layer, input)
        input = output
        chain.cacheList[index] = cache
    end

    # into loss
    loss, cache = Layer.forward(chain.lossLayer, input, y_true)
    chain.cacheList[end] = cache
    return loss
end

function backward(chain::Chain)
    grad_output, _ = Layer.backward(chain.lossLayer, chain.cacheList[end])
    for (layer, cache) in zip(reverse(chain.layers), reverse(chain.cacheList[1:end - 1]))

        grad_input, params = Layer.backward(layer, cache, grad_output)
        if params !== nothing
            layer.weights = layer.weights .- params[1] * chain.step
            layer.bias = layer.bias .- params[2] * chain.step
        end

        grad_output = grad_input
    end

    return grad_output
end

end