# mnist data dimension 784 * 60000
# Structure 784 - 500 - 250 - 100 - 30 - 100 - 250 - 500 - 784
# No need convolution, just need leaky rectlin transfer function
using Flux, MNIST
using Flux.Tracker

N=60000 # number of training points
D=784 # dimension of each x vector
H=100
batchSize = 1000
batchNum = convert(Int, N/batchSize)
iterations = 2
learning_rate = 0.2
k = 0.15
λ = 0.1 #for momentum

x, y = traindata()
x = x./255;
W1 = param(randn(H,D)./sqrt(D*H))
b1 = param(randn(H,1)./sqrt(H))
W2 = param(randn(D,H)./sqrt(D*H))
b2 = param(randn(D,1)./sqrt(D))

VdW1 = zeros(100,784)
Vdb1 = zeros(100,1)
VdW2 = zeros(784,100)
Vdb2 = zeros(784,1)

leakyReLU(x) = max.(0.01x, x)
loss(x) = sum((nnfun(x)- x).^2)/(D*N)
sigma(x) = 1./(1+exp.(-x))

function nnfun(x)
   out = sigma(W2*leakyReLU(W1*x.+b1).+b2)
   @show out
end

function E(bn)
  x_batch = x[:,(bn-1)*batchSize+1: bn*batchSize]
  loss(x_batch)
end

#Minibatch gradient descent
function update!(W1, b1, W2, b2, i, VdW1, Vdb1, VdW2, Vdb2, learning_rate)
   #eta = eta/(1+i*k) # learning rate annealing
   #@show VdW1, Vdb1, VdW2, Vdb2
   VdW1 = W1.grad * (1-λ) + VdW1 * λ
   Vdb1 = b1.grad * (1-λ) + Vdb1 * λ
   VdW2 = W2.grad * (1-λ) + VdW2 * λ
   Vdb2 = b2.grad * (1-λ) + Vdb2 * λ
   W1.data .-= VdW1 .* learning_rate
   b1.data .-= Vdb1 .* learning_rate
   W2.data .-= VdW2 .* learning_rate
   b2.data .-= Vdb2 .* learning_rate
   W1.grad = 0
   b1.grad .= 0
   W2.grad .= 0
   b2.grad .= 0
   return VdW1, Vdb1, VdW2, Vdb2
end


for i = 1:iterations
  for bn = 1:batchNum
    back!(E(bn))
    VdW1, Vdb1, VdW2, Vdb2 = update!(W1, b1, W2, b2, i, VdW1, Vdb1, VdW2, Vdb2, learning_rate)
    #@show E(bn)
  end
end
