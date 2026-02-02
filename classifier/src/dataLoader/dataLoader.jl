module DataLoader

using Random

const dataDir = joinpath(pwd(), "../data/")

const trainImagesPath = joinpath(dataDir, "train-images-idx3-ubyte")
const trainLabelsPath = joinpath(dataDir, "train-labels-idx1-ubyte")
const testImagesPath = joinpath(dataDir, "t10k-images-idx3-ubyte")
const testLabelsPath = joinpath(dataDir, "t10k-labels-idx1-ubyte")

include("./rawDataLoader.jl")

# types
export ClassifyImagesMatrix, ClassifyLabelsMatrix, DataBatch
# functions
export show_images_info, create_batches
# images and labels
export trainImages, testImages, trainLabels,testLabels 

trainImages = read_mnist_images(trainImagesPath)
testImages = read_mnist_images(testImagesPath)
trainLabels = read_mnist_labels(trainLabelsPath)
testLabels = read_mnist_labels(testLabelsPath)

function show_images_info(images::ClassifyImagesMatrix)
    println("Info for images: ")
    println("   numbers: ", images.num_images)
    println("   size: ", images.rows, "*", images.cols)
    println("   type: ", eltype(images.data))
    println("   range: [", minimum(images.data), " ,", maximum(images.data), "]")
end

function save_images_png(imgs::ClassifyImagesMatrix, fileDir::String, index::Int)
    img = imgs.data[:, :, index]
    filename = joinpath(fileDir, "test" * string(index) * ".png")
    img_norm = clamp.(img, 0, 1)
    img_grey = Gray.(img_norm)
    save(filename, img_grey)
    println("saved " * filename)
end

function create_batches(images::ClassifyImagesMatrix, labels::ClassifyLabelsMatrix,
    batch_size::Int=32, shuffle::Bool=true)
    num_samples = labels.num_labels
    indices = shuffle ? Random.shuffle(1:num_samples) : (1:num_samples)

    batches = []
    for begin_index in 1:batch_size:length(indices)
        batch_end = min(begin_index + batch_size - 1, num_samples)
        batch_indices = indices[begin_index:batch_end]

        batch_images = reshape(images.data[:, :, batch_indices], 
        (size(images.data, 1), size(images.data, 2), 1, length(batch_indices)))
        batch_labels = labels.data[batch_indices]

        push!(batches, DataBatch(batch_images, batch_labels, length(batch_indices), batch_indices))
    end

    return batches
end

end
