# cannot print visible digits
workspace()
using Flux, MNIST
using Flux.Tracker
using PyPlot
N = 50
D = 784
H = 100

(x, y) = traindata()
x = x./255
x = (x .- mean(x,1)) ./ std(x,1)
xtrain = x[:,1:N]

epsilon = 0.01 # for leakyReLU
batchSize = 5
batchNum = div(N, batchSize)
iterations = 1000
eta = 10 # learning rate
k = 0.1 # coefficients for annealing
momentum = 0.9


# randomly initiate weight and bias
W1 = param(randn(H,D)./sqrt(D*H))
w1 = param(randn(H,1)./sqrt(D))

U1 = param(randn(D,H)./sqrt(D*H))
u1 = param(randn(D,1)./sqrt(H))

vW1=zeros(size(W1))
vw1=zeros(size(w1))

vU1=zeros(size(U1))
vu1=zeros(size(u1))

loss(x) = sum((x - nnfun(x)).^2)/(N*D)
nnfun(x) = decode(encode(x))
sigma(x) = 1.0./(1.0 + exp.(-x))
leakyReLU(x) = max.(epsilon*x, x)
encode(x)= leakyReLU(W1*x .+ w1)
decode(h)= sigma(U1*h .+ u1)

#=
function nnfun(x)
   out = sigma(W2*leakyReLU(W1*x.+b1).+b2)
end
=#
function E(bn)
  x_batch = xtrain[:,(bn-1)*batchSize+1: bn*batchSize]
  loss(x_batch)
end


function update!(ps, vs, eta, i)
   eta = eta / (1+i*k)
   for count = 1: length(ps)
       copy!(vs[count], momentum * vs[count] + (1-momentum) * ps[count].grad)
       copy!(ps[count].data, ps[count].data - eta * vs[count])
   end
   for pars in ps
       pars.grad .= 0 # clear the gradient
   end
   return ps,vs,eta
end

for i = 1:iterations
   for bn = 1:batchNum
      back!(E(bn))
      ps,vs,eta = update!((W1, w1, U1, u1),(vW1,vw1,vU1,vu1),eta,i)
      @show E(bn)
   end
end

result = nnfun(xtrain).data
pcolormesh(reshape(result[:,1],28,28))
