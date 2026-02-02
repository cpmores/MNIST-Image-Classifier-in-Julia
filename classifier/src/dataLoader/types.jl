struct ClassifyImagesMatrix
    data::Array{Float32,3}
    rows::Int32
    cols::Int32
    num_images::Int32
    filename::String
end

struct ClassifyLabelsMatrix
    data::Array{UInt8}
    num_labels::Int32
    filename::String
end

struct DataBatch
    images::Array{Float32,4}
    labels::Array{UInt8}
    batch_size::Int
    indices::Array{Int}
end
