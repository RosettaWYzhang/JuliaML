# Julia simple neural net for Boston Housing Problem
using Flux, Flux.Tracker

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
iterations = 2000
learning_rate = 0.1

W1 = param(randn(hidden_units, D))
b1 = param([0.])
W2 = param(randn(1, hidden_units))
b2 = param([0.])

hiddenLayer(x) = W1*x .+ b1
outputLayer(x) = W2*x .+ b2
predictPrice(x) = outputLayer(sigma(hiddenLayer(x)))

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

for i = 1:iterations
  back!(E(x, y))
  update!((W1, b1, W2, b2))
  @show E(x, y)
end
