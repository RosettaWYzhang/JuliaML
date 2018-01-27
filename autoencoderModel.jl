using Flux #Flux.Data.MNIST
using Flux: onehotbatch, argmax, mse, throttle, accuracy
using Base.Iterators: partition
using Juno: @progress
using MNIST

trainX, trainY = traindata()
N = 60000
xtrain=trainX[:,1:N]./255
batchSize = 1000
batchNum = div(N,batchSize)

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

opt = Momentum(params(m)) # default η = 0.01; ρ = 0.9, decay = 0
squared_loss_test(x) = (sum(sum((x - m(x).data).^2))/batchSize)
log_loss(x) = begin; y=m(x); return -sum( x.*log.(y)+(1-x).*log.(1-y))/N; end

for i = 1:10
  info("Epoch $i")
  for bn = 1:batchNum
    info("batch $bn")
    x_batch = xtrain[:,(bn-1)*batchSize+1: bn*batchSize]
    Flux.train!(log_loss, x_batch, opt, cb = () -> @show squared_loss_test(x_batch))
    info("finish batch $bn")
  end
end
