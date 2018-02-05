# Mike's autoencoder code
using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, mse, throttle, accuracy
using Base.Iterators: partition
using Juno: @progress
using StatsFuns
using CuArrays

# Encode MNIST images as compressed vectors that can later be decoded back into
# images.
imgs = MNIST.images()
N = 60000 # Size of the encoding
batchSize = 200
# Partition into batches
X =  hcat(float.(reshape.(imgs, :))...)
data = [(float(hcat(vec.(imgs[i])...)),) for i in partition(1:60_000, batchSize)]
data = cu.(data)
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
  Dense(1000, 784)
  )
m = mapleaves(cu, m)

sigma(x)=1./(1+exp.(-x))
loss(x) = mse(m(x), x)
squared_loss_test(x) = sum(sum((x - sigma(m(x))).^2))/batchSize
log_loss(x) = begin; y=m(x); return -sum( x.*log.(y)+(1-x).*log.(1-y))/batchSize; end
# combine sigma function with log loss
wyloss(x) = begin; h=m(x); return mean(-x.*h + log1pexp.(h)); end
cuLoss(x) = begin; h=m(x); return mean(-x.*h + CUDAnative.log.(1+exp.(h))); end


evalcb = () -> @show squared_loss_test(data[1][1])
opt = Momentum(params(m),0.1)

@progress for i = 1:10
  info("Epoch $i")
  Flux.train!(cuLoss, data, opt, cb = evalcb)
end
