# Mike's sample code for autoencoder

using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, mse, throttle, accuracy
using Base.Iterators: partition
using Juno: @progress

# Encode MNIST images as compressed vectors that can later be decoded back into
# images.

imgs = MNIST.images()

# Partition into batches of size 1000
data = [(float(hcat(vec.(imgs[i])...)),) for i in partition(1:60_000, 1000)]

N = 32 # Size of the encoding

#=m = Chain(
  Dense(28^2, N, relu), # Encoder
  Dense(N, 28^2, relu)) # Decoder
=#
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
  
loss(x) = mse(m(x), x)

evalcb = throttle(() -> @show loss(data[1][1]), 5)
opt = ADAM(params(m))

@progress for i = 1:10
  info("Epoch $i")
  Flux.train!(loss, data, opt, cb = evalcb)
end

using Images

img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))

function sample()
  # 20 random digits
  before = [imgs[i] for i in rand(1:length(imgs), 20)]
  # Before and after images
  after = img.(map(x -> m(float(vec(x))).data, before))
  # Stack them all together
  hcat(vcat.(before, after)...)
end

cd(@__DIR__)

save("sample.png", sample())
