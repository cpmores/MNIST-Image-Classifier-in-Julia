module Layer

# export LayerCache
export AbstractLayer, ConvLayer, ReLU, MaxPoolLayer, Flatten, DenseLayer, SoftmaxWithCrossEntropyLoss
export forward, backward
export LayerCache

include("types.jl")
include("utils.jl")

using .LayerTypes
using .LayerUtils
using Dates

function forward(layer::ConvLayer, x::Array{Float32,3})
    output = conv_forward(x, layer.weights, layer.bias, layer.stribe, layer.padding)
    cache = LayerCache(x, output, Dict())
    return output, cache
end

function backward(layer::ConvLayer, cache::LayerCache, grad_output)
    # 1. get last layer's input delta
    # 1) shape a new output delta with stribe 
    stribe = layer.stribe
    grad_output_height = size(grad_output)[1]
    grad_output_width = size(grad_output)[2]
    kernel_height = size(layer.weights)[1]
    kernel_width = size(layer.weights)[2]

    num_output_channnels = size(cache.output)[3]
    num_input_channels = size(cache.input)[3]

    compute_output_height = grad_output_height + (grad_output_height - 1) * (stribe - 1)
    compute_output_width = grad_output_width + (grad_output_width - 1) * (stribe - 1)
    compute_output_height_padding = kernel_height - 1
    compute_output_width_padding = kernel_width - 1
    compute_output_delta = zeros(Float32, compute_output_height + 2 * compute_output_height_padding,
        compute_output_width + 2 * compute_output_width_padding, num_output_channnels)

    for noc in 1:num_output_channnels

        for goh in 1:grad_output_height

            for gow in 1:grad_output_width
                num = grad_output[goh, gow, noc]
                newHeight = (goh - 1) * stribe + 1 + compute_output_height_padding
                newWidth = (gow - 1) * stribe + 1 + compute_output_width_padding
                compute_output_delta[newHeight, newWidth, noc] = num
            end
        end
    end

    # 2) calculate input delta
    grad_input_height = compute_output_height + compute_output_height_padding
    grad_input_width = compute_output_width + compute_output_width_padding
    grad_input = zeros(Float32, grad_input_height, grad_input_width, num_input_channels)
    for noc in 1:num_output_channnels

        computeDelta = @view compute_output_delta[:, :, noc]
        for nic in 1:num_input_channels

            kernel = @view layer.weights[:, :, nic, noc]
            for gih in 1:grad_input_height

                for giw in 1:grad_input_width

                    nowComputeDelta = @view computeDelta[gih:gih+kernel_height-1, giw:giw+kernel_width-1]
                    for h in 1:kernel_height

                        for w in 1:kernel_width

                            grad_input[gih, giw, nic] += kernel[turn_kernel_around(h, w, kernel_height, kernel_width)...] * nowComputeDelta[h, w]
                        end
                    end
                end
            end
        end
    end

    # 2. get last layer's kernel delta
    kernelDelta = zeros(Float32, kernel_height, kernel_width, num_input_channels, num_output_channnels)
    # 3. get last layer's bias delta
    biasDelta = zeros(Float32, num_output_channnels)
    for noc in 1:num_output_channnels

        computeDelta =
            @view compute_output_delta[compute_output_height_padding+1:compute_output_height_padding+compute_output_height,
                compute_output_width_padding+1:compute_output_width_padding+compute_output_width,
                noc]
        for coh in 1:compute_output_height

            for cow in 1:compute_output_width

                biasDelta[noc] += computeDelta[coh, cow]
            end
        end

        for nic in 1:num_input_channels

            input = @view cache.input[:, :, nic]
            for kh in 1:kernel_height

                for kw in 1:kernel_width

                    computeInput =
                        @view input[kh:kh+compute_output_height-1,
                            kw:kw+compute_output_width-1]

                    for h in 1:compute_output_height

                        for w in 1:compute_output_width

                            kernelDelta[kh, kw, nic, noc] += computeInput[h, w] * computeDelta[h, w]
                        end
                    end
                end
            end
        end
    end

    return grad_input, (kernelDelta, biasDelta)

end

# for ReLU
function forward(layer::ReLU, x) 
    output = max.(0, x)
    cache = LayerCache(x, output, Dict())
    return output, cache
end

function backward(layer::ReLU, cache::LayerCache, grad_output)
    grad_input = similar(grad_output)

    for i in eachindex(grad_output)
        if cache.input[i] <= 0 
            grad_input[i] = 0
        else
            grad_input[i] = grad_output[i]
        end
    end

    return grad_input, nothing
end

# for max pooling
function forward(layer::MaxPoolLayer, x)
    outputHeight = size(x)[1] ÷ layer.size[1] 
    outputWidth = size(x)[2] ÷ layer.size[2]
    num_output_channels = size(x)[3]

    output = zeros(Float32, outputHeight, outputWidth, num_output_channels)
    maxPoint = zero(x)
    for noc in 1:num_output_channels

        for ph in 1:outputHeight
            
            leftHeadHeight = (ph - 1) * layer.size[1] + 1
            for pw in 1:outputWidth

                leftHeadWidth = (pw - 1) * layer.size[2] + 1
                range = @view x[ph:ph+layer.size[1]-1, pw:pw+layer.size[2]-1, noc]
                output[ph, pw, noc], maxIndex = findmax(range)

                maxIndexHeight = (maxIndex[1] - 1) + leftHeadHeight
                maxIndexWidth = (maxIndex[2] - 1) + leftHeadWidth
                maxPoint[maxIndexHeight, maxIndexWidth, noc] = 1.0f0
            end
        end
    end

    cache = LayerCache(x, output, Dict(:MaxPoint => maxPoint))
    return output, cache
end

function backward(layer::MaxPoolLayer, cache::LayerCache, grad_output)
    num_output_channels = size(cache.output)[3]
    grad_output_height = size(grad_output)[1]
    grad_output_width = size(grad_output)[2]
    grad_input_height = size(cache.input)[1]
    grad_input_width = size(cache.input)[2]
    poolHeight = layer.size[1]
    poolWidth = layer.size[2]

    grad_input = zero(cache.input)
    for noc in 1:num_output_channels

        maxPoint = @view cache.extra[:MaxPoint][1:grad_input_height, 1:grad_input_width, noc]
        grad_output_point = @view grad_output[1:grad_output_height, 1:grad_output_width, noc]
        for goh in 1:grad_output_height

            for gow in 1:grad_output_width

                grad = grad_output_point[goh, gow] 
                grad_input_height_start = (goh - 1) * poolHeight + 1
                grad_input_width_start = (gow - 1) * poolWidth + 1
                grad_input_height_end = grad_input_height_start + poolHeight - 1
                grad_input_width_end = grad_input_width_start + poolWidth - 1
                for h in grad_input_height_start:grad_input_height_end

                    for w in grad_input_width_start:grad_input_width_end
                        
                        grad_input[h, w, noc] = maxPoint[h, w] * grad 
                    end
                end
            end
        end
    end

    return grad_input, nothing
end

function forward(layer::Flatten, x)
    inputHeight, inputWidth, num_input_channels = size(x)
    eachChannel = inputHeight * inputWidth
    eachRow = inputWidth
    output = zeros(Float32, inputHeight * inputWidth * num_input_channels)
    for index in 1:size(output)[1]
        
        ch = (index - 1) ÷ eachChannel + 1
        restIndex1 = index - (ch - 1) * eachChannel
        height = (restIndex1 - 1) ÷ eachRow + 1
        width = restIndex1 - (height - 1) * eachRow
        output[index] = x[height, width, ch]
    end

    cache = LayerCache(x, output, Dict())
    return output, cache
end

function backward(layer::Flatten, cache::LayerCache, grad_output)
    inputHeight, inputWidth, num_input_channels = size(cache.input)
    eachChannel = inputHeight * inputWidth
    eachRow = inputWidth
    grad_input = zeros(Float32, inputHeight, inputWidth, num_input_channels)
    for index in 1:size(grad_output)[1]
        
        ch = (index - 1) ÷ eachChannel + 1
        restIndex1 = index - (ch - 1) * eachChannel
        height = (restIndex1 - 1) ÷ eachRow + 1
        width = restIndex1 - (height - 1) * eachRow
        grad_input[height, width, ch] = grad_output[index]
    end

    return grad_input, nothing
end


function forward(layer::DenseLayer, x)
    @assert size(layer.weights, 2) == length(x)
    
    output = layer.weights * x .+ layer.bias
    
    cache = LayerCache(x, output, Dict())
    return output, cache
end

function backward(layer::DenseLayer, cache::LayerCache, grad_output)
    grad_input = layer.weights' * grad_output
    
    grad_weights = grad_output * cache.input'
    
    grad_bias = copy(grad_output)
    return grad_input, (grad_weights, grad_bias)
end

function forward(layer::SoftmaxWithCrossEntropyLoss, x, y_true)
    x_max = maximum(x)
    exp_x = exp.(x .- x_max)
    sum_exp_x = sum(exp_x)
    probs = exp_x ./ sum_exp_x
    
    log_probs = log.(max.(probs, 1e-10))
    loss = -sum(y_true .* log_probs) / length(x)
    cache = LayerCache(x, loss, Dict(:Probs => probs, :y_true => y_true))
    
    return loss, cache
end

function backward(layer::SoftmaxWithCrossEntropyLoss, cache::LayerCache, grad_output=1.0f0)
    probs = cache.extra[:Probs]
    y_true = cache.extra[:y_true]
    batch_size = size(probs, 1)
    
    grad_input = probs - y_true
    grad_input = grad_input ./ batch_size .* grad_output 
    return grad_input, nothing
end

end