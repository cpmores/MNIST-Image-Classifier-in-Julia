using FileIO, Images
include("types.jl")

function read_mnist_images(filename)::ClassifyImagesMatrix
    open(filename) do io
        # get meta
        magic_number = ntoh(read(io, Int32))
        num_images = ntoh(read(io, Int32))
        img_rows = ntoh(read(io, Int32))
        img_cols = ntoh(read(io, Int32))

        # read real images
        images = Array{UInt8}(undef, img_cols, img_rows, num_images)
        read!(io, images)

        images_float = permutedims(images, (2, 1, 3))
        normalized = Float32.(images_float) ./ 255.0f0

        return ClassifyImagesMatrix(normalized, img_rows, img_cols, num_images, filename)
    end
end

function read_mnist_labels(filename)::ClassifyLabelsMatrix
    open(filename) do io
        magic_number = ntoh(read(io, Int32))
        num_labels = ntoh(read(io, Int32))
        labels = Array{UInt8}(undef, num_labels)
        read!(io, labels)
        return ClassifyLabelsMatrix(labels, num_labels, filename)
    end
end


