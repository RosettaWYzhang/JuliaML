# Mike's autoencoder code
using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, mse, throttle, accuracy
using Base.Iterators: partition
using Juno: @progress

# Encode MNIST images as compressed vectors that can later be decoded back into
# images.

imgs = MNIST.images()
N = 60000 # Size of the encoding
batchSize = 1000
# Partition into batches
data = [(float(hcat(vec.(imgs[i])...)),) for i in partition(1:60_000, batchSize)]

# define autoencoder structure: 784-1000-500-250-30
m = Chain(
  # Encoder
  Dense(784, 1000, leakyrelu),
  Dense(1000, 500, leakyrelu),
  Dense(500, 250, leakyrelu),
  Dense(250, 30, leakyrelu),
  # Decoder
  Dense(30, 250, leakyrelu),
  Dense(250, 500, leakyrelu),
  Dense(500, 1000, leakyrelu),
  Dense(1000, 784, σ))

squared_loss_test(x) = sum(sum((x - m(x)).^2))/batchSize
log_loss(x) = begin; y=m(x); return -sum( x.*log.(y)+(1-x).*log.(1-y))/batchSize; end

evalcb = () -> @show squared_loss_test(data[1][1])
opt = Momentum(params(m))

@progress for i = 1:10
  info("Epoch $i")
  Flux.train!(log_loss, data, opt, cb = evalcb)
end
