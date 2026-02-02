module classifier

include("../src/dataLoader/dataLoader.jl")
include("../src/layers/layers.jl")
include("../src/train/model.jl")
include("../src/train/test.jl")
include("../src/utils/utils.jl")

using .DataLoader
using .Layer
using .Model
using .Test
using .Utils

using ProgressMeter
using JLD2
using Dates

export ConvLayer, ReLU, MaxPoolLayer, Flatten, DenseLayer, SoftmaxWithCrossEntropyLoss
export train, save, load, isSaved
export Mod, Chain
export trainImages, trainLabels, testImages, testLabels

batches::Vector{DataBatch} = create_batches(trainImages, trainLabels)
testBatches::Vector{DataBatch} = create_batches(testImages, testLabels)

function train(mod::Mod)
    println("Begin Training at $(Dates.format(now(), "HH:MM:SS"))")
    println("Total $(length(batches)) batches")
    
    start_time = time()
    
    for (batchIndex, batch) in enumerate(batches)
        batch_size = batch.batch_size
        @showprogress 1 "Training $(batchIndex)" for i in 1:batch_size
            image = batch.images[:, :, :, i]
            label = batch.labels[i]
            Model.forward(mod.chain, image, label)
            Model.backward(mod.chain)
        end
    end
    
    total_time = time() - start_time
    
    println("\n" * "="^50)
    println("Training Completed!")
    println("End Time: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println("  Total Training Time: $(format_time(total_time))")
    println("="^50)
end

function test(mod::Mod)
    println("Begin Testing at $(Dates.format(now(), "HH:MM:SS"))")
    println("Total $(length(testBatches)) batches")
    
    start_time = time()
    testResult = TestResult() 
    
    for (batchIndex, batch) in enumerate(testBatches)
        batch_size = batch.batch_size
        @showprogress 1 "Training $(batchIndex)" for i in 1:batch_size
            image = batch.images[:, :, :, i]
            label = batch.labels[i]
            indice = batch.indices[i]
            y_true, y_pred = Test.predict(mod.chain, image, label)
            updateResult(testResult, indice, y_true, y_pred)
        end
    end
    csvFile = joinpath(pwd(), "save/cnn_test_result.csv")
    save_to_csv(testResult, csvFile)
    
    total_time = time() - start_time
    
    println("\n" * "="^50)
    println("Testing Completed!")
    println("End Time: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println("  Total Testing Time: $(format_time(total_time))")
    println("="^50)
end

saveFile = joinpath(pwd(), "save/cnn_model.jld2")
function save(mod::Mod)
    @save saveFile mod
    println("Successfully saved $(saveFile)")
end

function isSaved()::Bool 
    return isfile(saveFile)
end

function load()
    if !isfile(saveFile)
        error("Checkpoint file not found: ", saveFile)
    end

    @load saveFile mod
    return mod
end


end # module classifier
