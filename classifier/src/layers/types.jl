module LayerTypes

export AbstractLayer, LayerCache
export forward, backward
export ConvLayer, ReLU, MaxPoolLayer, Flatten, DenseLayer, SoftmaxWithCrossEntropyLoss

abstract type AbstractLayer end

mutable struct LayerCache
    input
    output
    extra::Dict{Symbol,Any}

    function LayerCache(input=nothing, output=nothing, extra=Dict())
        new(input, output, extra)
    end
end

# needs implement
forward(layer::AbstractLayer, x) = error("Not implemented for $(typeof(layer))")
backward(layer::AbstractLayer, cache::LayerCache, grad_output) =
    error("Not implemented for $(typeof(layer))")

mutable struct ConvLayer <: AbstractLayer
    weights
    bias
    stribe::Int
    padding::Int
    kernel_size::Tuple{Int,Int}


    function ConvLayer(num_in_channel::Int, num_out_channel::Int, stribe::Int=1,
        padding::Int=0, kernel_size::Tuple{Int,Int}=(2, 2))::ConvLayer
        weight = randn(Float32, kernel_size..., num_in_channel, num_out_channel)
        bias = zeros(Float32, num_out_channel)
        new(weight, bias, stribe, padding, kernel_size)
    end
end

struct ReLU <: AbstractLayer
end

struct MaxPoolLayer <: AbstractLayer
    size
end

struct Flatten <: AbstractLayer
end

mutable struct DenseLayer <: AbstractLayer
    weights
    bias

    function DenseLayer(num_input_channels::Int, num_output_channels::Int)
        weights = randn(Float32, num_output_channels, num_input_channels)
        bias = zeros(Float32, num_output_channels)
        new(weights, bias)
    end
end

struct SoftmaxWithCrossEntropyLoss <: AbstractLayer
end

end
