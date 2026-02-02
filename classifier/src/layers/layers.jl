module Layer

# export LayerCache
export AbstractLayer, ConvLayer, ReLU, MaxPoolLayer, Flatten, DenseLayer, SoftmaxWithCrossEntropyLoss
export forward, backward
export LayerCache

include("types.jl")
include("utils.jl")

using .LayerTypes
using .LayerUtils

function forward(layer::ConvLayer, x::Array{Float32,3})
    output = conv_forward(x, layer.weights, layer.bias, layer.stribe, layer.padding)
    cache = LayerCache(x, output, Dict())
    return output, cache
end

function backward(layer::ConvLayer, cache::LayerCache, grad_output)
    # 1. get last layer's input delta
    
    # 1) shape a new output delta with stribe 
    stribe = layer.stribe
    H_grad, W_grad, C_out = size(grad_output)
    H_k, W_k, C_in, _ = size(layer.weights)
    
    # Compute expanded gradient dimensions
    H_expand = H_grad + (H_grad - 1) * (stribe - 1)
    W_expand = W_grad + (W_grad - 1) * (stribe - 1)
    pad_h = H_k - 1
    pad_w = W_k - 1
    
    # Initialize expanded gradient with zeros
    grad_expanded = zeros(Float32, H_expand + 2pad_h, W_expand + 2pad_w, C_out)
    
    # Insert gradients at strided positions
    @inbounds for oc in 1:C_out
        for h in 1:H_grad
            h_exp = (h - 1) * stribe + 1 + pad_h
            for w in 1:W_grad
                w_exp = (w - 1) * stribe + 1 + pad_w
                grad_expanded[h_exp, w_exp, oc] = grad_output[h, w, oc]
            end
        end
    end
    
    # 2) calculate input delta
    H_in_grad = H_expand + pad_h
    W_in_grad = W_expand + pad_w
    grad_input = zeros(Float32, H_in_grad, W_in_grad, C_in)
    
    @inbounds for oc in 1:C_out
        grad_exp = @view grad_expanded[:, :, oc]
        for ic in 1:C_in
            kernel = @view layer.weights[:, :, ic, oc]
            # Rotate kernel 180 degrees for convolution
            for h in 1:H_in_grad
                for w in 1:W_in_grad
                    patch = @view grad_exp[h:h+H_k-1, w:w+W_k-1]
                    acc = 0.0f0
                    for kh in 1:H_k, kw in 1:W_k
                        # Flip kernel indices
                        k_rot_h = H_k - kh + 1
                        k_rot_w = W_k - kw + 1
                        acc += kernel[k_rot_h, k_rot_w] * patch[kh, kw]
                    end
                    grad_input[h, w, ic] += acc
                end
            end
        end
    end
    
    # 2. get last layer's kernel delta
    kernelDelta = zeros(Float32, H_k, W_k, C_in, C_out)
    
    # 3. get last layer's bias delta
    biasDelta = zeros(Float32, C_out)
    
    # Extract valid region of expanded gradient (without padding)
    grad_valid = @view grad_expanded[pad_h+1:pad_h+H_expand, pad_w+1:pad_w+W_expand, :]
    
    @inbounds for oc in 1:C_out
        grad_slice = @view grad_valid[:, :, oc]
        
        # Compute bias gradient (sum of all gradient elements)
        bias_acc = 0.0f0
        for h in 1:H_expand, w in 1:W_expand
            bias_acc += grad_slice[h, w]
        end
        biasDelta[oc] = bias_acc
        
        # Compute weight gradients
        for ic in 1:C_in
            input_slice = @view cache.input[:, :, ic]
            for kh in 1:H_k, kw in 1:W_k
                # Extract input patch for this kernel position
                input_patch = @view input_slice[kh:kh+H_expand-1, kw:kw+W_expand-1]
                
                acc = 0.0f0
                for h in 1:H_expand, w in 1:W_expand
                    acc += input_patch[h, w] * grad_slice[h, w]
                end
                kernelDelta[kh, kw, ic, oc] = acc
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
    @assert size(layer.weights)[2] == size(x)[1] 
    num_output_channels = size(layer.weights)[1]
    num_input_channels = size(layer.weights)[2]
    output = zeros(Float32, num_output_channels)
    
    for noc in 1:num_output_channels
        # Initialize with bias and accumulate weighted inputs
        output[noc] = layer.bias[noc]
        weight_line = @view layer.weights[noc, :]
        
        # Dot product between weight row and input
        acc = 0.0f0
        @simd for nic in 1:num_input_channels
            acc += weight_line[nic] * x[nic]
        end
        output[noc] += acc
    end

    cache = LayerCache(x, output, Dict())
    return output, cache
end

function backward(layer::DenseLayer, cache::LayerCache, grad_output)
    num_output_channels = size(grad_output, 1)
    num_input_channels = length(cache.input)
    
    # 1. Compute input gradient: W^T * grad_output
    grad_input = zeros(Float32, num_input_channels)
    
    # Transposed matrix-vector multiplication
    for noc in 1:num_output_channels
        grad_val = grad_output[noc]
        weight_row = @view layer.weights[noc, :]
        
        # Accumulate contributions to each input neuron
        @simd for nic in 1:num_input_channels
            grad_input[nic] += weight_row[nic] * grad_val
        end
    end
    
    # 2. Compute weight gradients: grad_output * input^T
    grad_weights = zeros(Float32, num_output_channels, num_input_channels)
    
    # Outer product between grad_output and input
    for noc in 1:num_output_channels
        grad_val = grad_output[noc]
        @simd for nic in 1:num_input_channels
            grad_weights[noc, nic] = grad_val * cache.input[nic]
        end
    end
    
    # 3. Compute bias gradient (grad_output itself)
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
