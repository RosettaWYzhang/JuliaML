# Train an MNIST autoencoder
# Structure 784 - 500 - 250 - 100 - 30 - 100 - 250 - 500 - 784
# No need convolution, just need leaky rectlin transfer function
using Flux, MNIST
layerUnits = [784, 500, 250, 100, 30, 100, 250, 500, 784]
prev = 784
ϵ = 0.001


# function for creating linear layer
function linear(in, out)
  W = param(randn(out, in))
  b = param(randn(out))
  x -> W * x .+ b
end

for (index,value) in enumerate(layerUnits)
  layer[index] = linear(pre, value)
  prev = value
end

hiddenLayer = linear(D, hidden_units)
outputLayer = linear(hidden_units,1)


leakyReLU(x, ϵ) = (x < 0)? ϵ*x : x
