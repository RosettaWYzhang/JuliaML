# last layer: sigmoid
# all previous layer: leaky Relu
# minibatch, whole dataset
# Structure 784 - 500 - 250 - 100 - 30 - 100 - 250 - 500 - 784
# see how squared error compared with -log

# understand prove integration of gaussian probability density
# prove integration of p(x) x dx = mu
# will ask a related question, read about probability
# trick: log error function and sigmoid, a standard trick implemented in package

using PyPlot
using MNIST
using Flux.Tracker
(trainX, trainY) = traindata()

# intialise parameters
N = 50 # number of training points
D = 784 # dimension of each x vector
numUnits = [784,500,250,100,30]
numLayers = 8
xtrain = trainX[:,1:N]./255


# utility functions
function sigma(x)
    return 1./(1.0+exp.(-x))
end

function leakyReLU(x,alpha=0.1)
    return max.(alpha,x)
end

# define structure of linear layer
mutable struct Affine
  W
  b
  vW
  vb
end

Affine(in::Integer, out::Integer) =
  Affine(param(randn(out, in)), param(randn(out)), zeros(randn(out, in)), zeros(randn(out)))

# Overload call, so the object can be used as a function
(m::Affine)(x) = m.W * x .+ m.b

# autoencoder structure
en_l1 = Affine(numUnits[1],numUnits[2])
en_l2 = Affine(numUnits[2],numUnits[3])
en_l3 = Affine(numUnits[3],numUnits[4])
en_l4 = Affine(numUnits[4],numUnits[5])
de_l1 = Affine(numUnits[5],numUnits[4])
de_l2 = Affine(numUnits[4],numUnits[3])
de_l3 = Affine(numUnits[3],numUnits[2])
de_l4 = Affine(numUnits[2],numUnits[1])

layers = [en_l1, en_l2, en_l3, en_l4, de_l1, de_l2, de_l3, de_l4]

function clearGradient(layers)
    for i = 1 : numLayers
        layers[i].W.grad .= 0
        layers[i].b.grad .= 0
    end
end

encode(x) = leakyReLU(en_l4(leakyReLU(en_l3(leakyReLU(en_l2(leakyReLU(en_l1(x))))))))
decode(x) = sigma(de_l4(leakyReLU(de_l3(leakyReLU(de_l2(leakyReLU(de_l1(x))))))))
model(x) = decode(encode(x))
loss(x) = begin; y=model(x); return -sum( x.*log.(y)+(1-x).*log.(1-y))/(N*D); end

# update with momentum
function update!(layers,mu=0.9, eta = 5.5)
    for i = 1:numLayers
        layers[i].vW = mu * layers[i].vW + (1-mu) * layers[i].W.grad
        layers[i].vb = mu * layers[i].vb + (1-mu) * layers[i].b.grad
        layers[i].W.data = layers[i].W.data - eta * layers[i].vW
        layers[i].b.data = layers[i].b.data - eta * layers[i].vb
    end
    clearGradient(layers)
end

for i = 1:20
    back!(loss(xtrain))
    update!(layers)
    @show loss(xtrain)
end

xrecon = model(xtrain).data
pcolormesh(reshape(xrecon[:,1],28,28),cmap="gray")
