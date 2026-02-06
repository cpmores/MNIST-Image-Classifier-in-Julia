using Pkg
Pkg.activate(pwd())
Pkg.instantiate()
include("../src/classifier.jl")
using .classifier

if isfile(saveFile)
    mod = load()
else
    mod = Mod(Chain(
        ConvLayer(1, 32, 1, 1, (3, 3)),  # 28x28x1 -> 28x28x32
        ReLU(),
        MaxPoolLayer((2, 2)),            # 28x28x32 -> 14x14x32
        
        ConvLayer(32, 64, 1, 1, (3, 3)), # 14x14x32 -> 14x14x64
        ReLU(),
        MaxPoolLayer((2, 2)),            # 14x14x64 -> 7x7x64
        
        Flatten(),                       # 7x7x64 = 3136
        
        DenseLayer(3136, 128),
        ReLU(),
        DenseLayer(128, 10)
    ))

    train(mod)
    save(mod)
end

test(mod)


greet() = print("Hello World!")

greet()