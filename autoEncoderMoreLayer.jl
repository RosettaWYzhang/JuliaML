# V Structure 784 - 500 - 250 - 100 - 30 - 100 - 250 - 500 - 784
# last layer: sigmoid, all previous layer leaky relu
# use minibatch and whole dataset
# see how squared error compared with -log
# problem with this code: digit is not clear; loss decreases very slowly

# 3
# 1.4
# bad:14
# JLD
# put layer 1000
# try Mike's code
using PyPlot
using MNIST
using Flux.Tracker
trainX, trainY = traindata()

function sigma(x)
    return 1./(1.0+exp.(-x))
end

function leakyReLU(x, alpha=0.1)
    return max.(alpha,x)
end

N =60000 # number of training points
D = 784 # dimension of each x vector
H0 = 1000
H1 = 500
H2 = 250
H3 = 100
H4 = 30
batchSize = 100
batchNum = div(N,batchSize)
iteration = 2
learning_rate = 10
k = 0.01  #for learning_rate annealing

xtrain=trainX[:,1:N]./255

# randomly initiate weight and bias
# 784 * 1000
W1 = param(randn(H0,D)./sqrt(D))
w1 = param(randn(H0,1)./sqrt(H0))
# 1000 * 500
W2 = param(randn(H1,H0)./sqrt(D))
w2 = param(randn(H1,1)./sqrt(H1))
# 500 * 250
W3 = param(randn(H2,H1)./sqrt(H1))
w3 = param(randn(H2,1)./sqrt(H2))
# 250 * 100
W4 = param(randn(H3,H2)./sqrt(H2))
w4 = param(randn(H3,1)./sqrt(H3))
# 100 * 30
W5 = param(randn(H4,H3)./sqrt(H3))
w5 = param(randn(H4,1)./sqrt(H4))

# 30 * 100
U1 = param(randn(H3,H4)./sqrt(H3))
u1 = param(randn(H3,1)./sqrt(H3))
# 100 * 250
U2 = param(randn(H2,H3)./sqrt(H3))
u2 = param(randn(H2,1)./sqrt(H2))
# 250 * 500
U3 = param(randn(H1,H2)./sqrt(H2))
u3 = param(randn(H1,1)./sqrt(H1))
# 500 * 1000
U4 = param(randn(H0,H1)./sqrt(H1))
u4 = param(randn(H0,1)./sqrt(D))
# 1000 * 784
U5 = param(randn(D,H0)./sqrt(H0))
u5 = param(randn(D,1)./sqrt(D))


vW1=zeros(size(W1))
vw1=zeros(size(w1))
vW2=zeros(size(W2))
vw2=zeros(size(w2))
vW3=zeros(size(W3))
vw3=zeros(size(w3))
vW4=zeros(size(W4))
vw4=zeros(size(w4))
vW5=zeros(size(W5))
vw5=zeros(size(w5))

vU1=zeros(size(U1))
vu1=zeros(size(u1))
vU2=zeros(size(U2))
vu2=zeros(size(u2))
vU3=zeros(size(U3))
vu3=zeros(size(u3))
vU4=zeros(size(U4))
vu4=zeros(size(u4))
vU5=zeros(size(U5))
vu5=zeros(size(u5))

encode(x)= leakyReLU(W5*leakyReLU(W4*leakyReLU(W3*leakyReLU(W2*leakyReLU(W1*x .+ w1).+w2).+w3).+w4).+w5)
decode(h)= sigma(U5*leakyReLU(U4*leakyReLU(U3*leakyReLU(U2*leakyReLU(U1*h .+ u1).+u2).+u3).+u4).+u5)


#squared loss
#square_loss(x) = sum((x - decode(encode(x))).^2)/(N*D)
square_loss_test(x) = sum(sum((x - decode(encode(x)).data).^2))/N

loss(x) = begin; y=decode(encode(x)); return -sum( x.*log.(y)+(1-x).*log.(1-y))/(N*D); end

# compute minibatch loss
function E(bn)
  x_batch = xtrain[:,(bn-1)*batchSize+1: bn*batchSize]
  loss(x_batch)
end

# momentum
function momentum!(ps, vs, i, mu=0.9)
   eta = learning_rate/(1+k*i)
   for bn = 1 : batchNum
        back!(E(bn))
        @show square_loss_test(xtrain)
        for count=1:length(ps)
            copy!(vs[count],  mu*vs[count] + (1-mu)*ps[count].grad)
            copy!(ps[count].data, ps[count].data -eta*vs[count])
        end
        for pars in ps
            pars.grad .= 0 # clear the gradient
        end
    end
end

function update!()
    for i = 1:iteration
        momentum!((W1, w1, U1, u1, W2, w2, U2, u2, W3, w3, U3, u3, W4, w4, U4, u4,W5, w5, U5, u5),(vW1,vw1,vU1,vu1,vW2,vw2,vU2,vu2,vW3,vw3,vU3,vu3,vW4,vw4,vU4,vu4,vW5,vw5,vU5,vu5),i)
    end
end

update!()
xrecon=decode(encode(xtrain)).data
pcolormesh(reshape(xrecon[:,1],28,28),cmap="gray")
