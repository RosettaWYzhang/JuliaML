workspace()
using Flux, MNIST
using Flux.Tracker
using PyPlot
N = 50
D = 784
H = 100
(x, y) = traindata()
x = x./255;
x = (x .- mean(x,1)) ./ std(x,1)
x = x[:,1:N]
iterations = 1000
epsilon = 0.01
batchSize = 5
batchNum = div(N, batchSize)
learning_rate = 10
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

loss(x) = sum((nnfun(x)- x).^2)/(D*N)
sigma(x) = 1./(1+exp.(-x))
leakyReLU(x) = max.(epsilon*x, x)

function nnfun(x)
   out = sigma(W2*leakyReLU(W1*x.+b1).+b2)
end

function E(bn)
  x_batch = x[:,(bn-1)*batchSize+1: bn*batchSize]
  loss(x_batch)
end

function update!(W1, b1, W2, b2, VdW1, Vdb1, VdW2, Vdb2)
   VdW1 = W1.grad * (1-momentum) + VdW1 * momentum
   Vdb1 = b1.grad * (1-momentum) + Vdb1 * momentum
   VdW2 = W2.grad * (1-momentum) + VdW2 * momentum
   Vdb2 = b2.grad * (1-momentum) + Vdb2 * momentum
   W1.data .-= VdW1 .* learning_rate
   b1.data .-= Vdb1 .* learning_rate
   W2.data .-= VdW2 .* learning_rate
   b2.data .-= Vdb2 .* learning_rate
   W1.grad .= 0
   b1.grad .= 0
   W2.grad .= 0
   b2.grad .= 0
   return VdW1, Vdb1, VdW2, Vdb2
end


for i = 1:iterations
   for bn = 1:batchNum
      back!(E(bn))
      VdW1, Vdb1, VdW2, Vdb2 = update!(W1, b1, W2, b2, VdW1, Vdb1, VdW2, Vdb2)
      @show E(bn)
   end
end

result = nnfun(x).data
pcolormesh(reshape(result[:,1],28,28))
