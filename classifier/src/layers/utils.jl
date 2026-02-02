module LayerUtils
using ..LayerTypes

export conv_forward, turn_kernel_around

function conv_forward(input::Array{Float32,3}, weights::Array{Float32,4},
    bias::Vector{Float32}, stribe::Int, padding::Int)

    # 1. get size 
    input_height, input_width, input_channels = size(input)
    kernel_height, kernel_width, _, output_channels = size(weights)

    # 2. calculate output size 
    output_height = div(input_height + 2 * padding - kernel_height, stribe) + 1
    output_width = div(input_width + 2 * padding - kernel_width, stribe) + 1

    output = zeros(Float32, output_height, output_width, output_channels)

    # 3. add padding
    if padding > 0
        padded_input = zeros(Float32,
            input_height + 2 * padding,
            input_width + 2 * padding,
            input_channels)
        padded_input[padding+1:padding+input_height,
            padding+1:padding+input_width, :] .= input
        input = padded_input
        input_height, input_width, _ = size(input)
    end

    # 4. conv calculation
    for out_ch in 1:output_channels
        bias_val = bias[out_ch]

        for in_ch in 1:input_channels
            kernel = @view weights[:, :, in_ch, out_ch]

            for out_h in 1:output_height
                input_h = (out_h - 1) * stribe + 1

                for out_w in 1:output_width
                    input_w = (out_w - 1) * stribe + 1

                    input_block = @view input[input_h:input_h+kernel_height-1,
                        input_w:input_w+kernel_width-1,
                        in_ch]

                    conv_val = 0.0f0
                    for kh in 1:kernel_height, kw in 1:kernel_width
                        conv_val += input_block[kh, kw] * kernel[kh, kw]
                    end

                    if in_ch == 1
                        output[out_h, out_w, out_ch] = bias_val + conv_val
                    else
                        output[out_h, out_w, out_ch] += conv_val
                    end
                end
            end
        end
    end

    return output
end

function turn_kernel_around(h, w, height, width)
    return height + 1 - h, width + 1 - w
end

end  # module
