# Julia simple neural net for Boston Housing Problem
# f(x) = c'h(x)
# h(x) = σ(Ax + b)
workspace()
using Flux, Flux.Tracker
#using CuArrays

isfile("housing.data") ||
  download("https://raw.githubusercontent.com/MikeInnes/notebooks/master/housing.data",
           "housing.data")

rawdata = readdlm("housing.data")'

# The last feature is our target -- the price of the house.
x = rawdata[1:13,:]
y = rawdata[14:14,:]
# Normalise the data
x = (x .- mean(x,2)) ./ std(x,2)

N = 505 # number of training points
D = 13 # dimension of each x vector
hidden_units = 20
iterations = 5000
learning_rate = 0.1

# function for creating linear layer
function linear(in, out)
  W = param(randn(out, in))
  b = param(randn(out))
  x -> W * x .+ b
end

hiddenLayer = linear(D, hidden_units)
outputLayer = linear(hidden_units,1)

sigma(x) = 1./(1.0+exp.(-x))
meansquareloss(yhat, y) = sum((yhat - y).^2)/N
E(x, y) = meansquareloss(predictPrice(x), y)
predictPrice(x) = outputLayer(sigma(hiddenLayer(x)))
function update!(ps, eta = .1)
  for pars in ps
    pars.data .-= pars.grad .* eta
    pars.grad .= 0
  end
end

#GPU support
#x,y = cu(x),cu(y)
#hiddenLayer = mapleaves(cu, hiddenLayer)
#outputLayer = mapleaves(cu, outputLayer)

for i = 1:iterations
  back!(E(x, y))
  update!((hiddenLayer.W, hiddenLayer.b, outputLayer.W, outputLayer.b))
  @show E(x, y)
end
