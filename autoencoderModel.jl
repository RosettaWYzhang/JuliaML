# mnist data dimension 784 * 60000
# Structure 784 - 500 - 250 - 100 - 30 - 100 - 250 - 500 - 784
# No need convolution, just need leaky rectlin transfer function
using Flux, MNIST
using Flux: onehotbatch, argmax, mse, throttle
using Base.Iterators: repeated

x, y = traindata()
y = onehotbatch(y, 0:9)


leakyReLU(x) = max(0.01x, x)
loss(x, y) = mse(m(x), y)

m = Chain(
   Dense(784, 500, leakyReLU),
   Dense(500, 250, leakyReLU),
   Dense(250, 100, leakyReLU),
   Dense(100, 30, leakyReLU),
   Dense(30, 100, leakyReLU),
   Dense(100, 250, leakyReLU),
   Dense(250, 500, leakyReLU),
   Dense(500, 784),
   softmax)
#=
encoder = Chain(
   Dense(784, 500, leakyrelu),
   Dense(500, 250, leakyrelu),
   Dense(250, 100, leakyrelu),
   Dense(100, 30, leakyrelu)
)

decoder = Chain(
  Dense(30, 100, leakyrelu),
  Dense(100, 250, leakyrelu),
  Dense(250, 500, leakyrelu),
  Dense(500, 784),
  softmax
)

autoencoder = Chain(
   encoder,
   decoder
)
=#


dataset = repeated((x, y), 1)
evalcb = () -> @show(loss(x, y))
opt = SGD(params(m), 0.1)

Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 5))

# Check the prediction for the first digit
argmax(m(x[:,1]), 0:9) == argmax(y[:,1], 0:9)
