# Train an MNIST autoencoder
# Structure 784 - 500 - 250 - 100 - 30 - 100 - 250 - 500 - 784
# No need convolution, just need leaky rectlin transfer function
using Flux, MNIST
using Flux.Tracker
N = 60000
D = 784
H = 100
x, y = traindata()
x = (x .- mean(x,1)) ./ std(x,1)
# Normalise the data
#x = x./255;  # x dimension: 784 * 60000
iterations = 50
epsilon = 0.01
batchSize = 1000
batchNum = div(N, batchSize)
iterations = 50
learning_rate = 1
eta = learning_rate
momentum = 0.9


W1 = param(randn(H,D)./sqrt(D*H))
b1 = param(randn(H,1)./sqrt(H))
W2 = param(randn(D,H)./sqrt(D*H))
b2 = param(randn(D,1)./sqrt(D))

VdW1 = zeros(100,784)
Vdb1 = zeros(100,1)
VdW2 = zeros(784,100)
Vdb2 = zeros(784,1)

gamma = param(randn(100,1))
beta = param(randn(100,1))


loss(x) = sum((nnfun(x)- x).^2)/(D*N)
sigma(x) = 1./(1+exp.(-x))
leakyReLU(x) = max.(epsilon*x, x)
batchNorm(x,gamma,beta) = gamma*(x .- mean(x,1)) ./ std(x,1) + beta


function nnfun(x)
   out = sigma(W2*leakyReLU(W1*x.+b1).+b2)
end

function E(bn,gamma,beta)
  x_batch = x[:,(bn-1)*batchSize+1: bn*batchSize]
  x_batch = batchNorm(x_batch,gamma,beta)
  loss(x_batch)
end

#=
function update!(ps,learning_rate=0.2, momentum = 0.9)
   for pars in ps
      pars.data .-= pars.grad .* learning_rate
      pars.grad .= 0
   end
end
=#

function update!(W1, b1, W2, b2, VdW1, Vdb1, VdW2, Vdb2)
   #eta = eta/(1+i*learning_rate) # learning rate annealing
   VdW1 = W1.grad * (1-momentum) + VdW1 * momentum
   Vdb1 = b1.grad * (1-momentum) + Vdb1 * momentum
   VdW2 = W2.grad * (1-momentum) + VdW2 * momentum
   Vdb2 = b2.grad * (1-momentum) + Vdb2 * momentum
   gamma .-= gamma.grad * learning_rate
   beta .-= beta.grad * learning_rate
   W1.data .-= VdW1 .* learning_rate
   b1.data .-= Vdb1 .* learning_rate
   W2.data .-= VdW2 .* learning_rate
   b2.data .-= Vdb2 .* learning_rate
   W1.grad .= 0
   b1.grad .= 0
   W2.grad .= 0
   b2.grad .= 0
   return VdW1, Vdb1, VdW2, Vdb2, gamma, beta
end


for i = 1:iterations
   for bn = 1:batchNum
      back!(E(bn,gamma,beta))
      VdW1, Vdb1, VdW2, Vdb2, gamma, beta = update!(W1, b1, W2, b2, VdW1, Vdb1, VdW2, Vdb2,gamma,beta)
      @show E(bn,gamma,beta)
   end
end
