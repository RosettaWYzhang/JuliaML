# Train an MNIST autoencoder
# Structure 784 - 500 - 250 - 100 - 30 - 100 - 250 - 500 - 784
# No need convolution, just need leaky rectlin transfer function
using Flux, MNIST
using Flux.Tracker
N = 1000
D = 784
H = 100
x, y = traindata()
x = x[:, 1:1000]
x = x./255;
iterations = 50
learning_rate = 5
epsilon = 0.01

W1 = param(randn(H,D)./sqrt(D*H))
b1 = param(randn(H,1)./sqrt(H))
W2 = param(randn(D,H)./sqrt(D*H))
b2 = param(randn(D,1)./sqrt(D))

loss(x) = sum((nnfun(x)- x).^2)/(D*N)
sigma(x) = 1./(1+exp.(-x))
leakyReLU(x) = max.(epsilon*x, x)


function nnfun(x)
   out = sigma(W2*leakyReLU(W1*x.+b1).+b2)
end

function update!(ps,learning_rate=0.2)
   for pars in ps
      pars.data .-= pars.grad .* learning_rate
      pars.grad .= 0
   end
end


for i = 1:iterations
  back!(loss(x))
  update!((W1, b1, W2, b2))
  @show loss(x)
end
