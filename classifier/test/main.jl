using Pkg
Pkg.activate(pwd())
Pkg.instantiate()
include("../src/classifier.jl")
using .classifier

if isSaved()
    mod = load()
else
    mod = Mod(Chain(
        ConvLayer(1, 32, 2, 0, (2, 2)),
        ReLU(),
        MaxPoolLayer((2, 2)),
        Flatten(),
        DenseLayer(trainImages.rows * trainImages.cols * 32 ÷ 16, 4096),
        ReLU(),
        DenseLayer(4096, 256),
        ReLU(),
        DenseLayer(256, 10)
    ))
end

train(mod)
save(mod)
test(mod)

greet() = print("Hello World!")

greet()