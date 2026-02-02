module LayerUtils
using ..LayerTypes

export conv_forward, turn_kernel_around

function conv_forward(input::Array{Float32,3}, weights::Array{Float32,4},
                      bias::Vector{Float32}, stride::Int, padding::Int)
    
    H_in, W_in, C_in = size(input)
    H_k, W_k, _, C_out = size(weights)
    
    H_out = div(H_in + 2 * padding - H_k, stride) + 1
    W_out = div(W_in + 2 * padding - W_k, stride) + 1
    
    if padding > 0
        padded = zeros(Float32, H_in + 2padding, W_in + 2padding, C_in)
        padded[padding+1:padding+H_in, padding+1:padding+W_in, :] .= input
        input = padded
        H_in, W_in, _ = size(input)
    end
    
    output = zeros(Float32, H_out, W_out, C_out)
    
    @inbounds for oc in 1:C_out
        b = bias[oc]
        for h_out in 1:H_out
            h_in = (h_out - 1) * stride + 1
            for w_out in 1:W_out
                w_in = (w_out - 1) * stride + 1
                acc = 0.0f0
                for ic in 1:C_in
                    kernel = @view weights[:, :, ic, oc]
                    patch = @view input[h_in:h_in+H_k-1, w_in:w_in+W_k-1, ic]
                    for kh in 1:H_k, kw in 1:W_k
                        acc += patch[kh, kw] * kernel[kh, kw]
                    end
                end
                output[h_out, w_out, oc] = b + acc
            end
        end
    end
    
    return output
end

function turn_kernel_around(h, w, height, width)
    return height + 1 - h, width + 1 - w
end

end  # module
