# V last layer: sigmoid
# V all previous layer: leaky Relu
# minibatch, whole dataset
# V Structure 784 - 500 - 250 - 100 - 30 - 100 - 250 - 500 - 784
# see how squared error compared with -log

# understand prove integration of gaussian probability density
# prove integration of p(x) x dx = mu, isn't this the definition?
# will ask a related question, read about probability
# trick: log error function and sigmoid, a standard trick implemented in package

# Structure 784 - 250 - 20 - 250 - 784

using PyPlot
using MNIST
using Flux.Tracker
(trainX, trainY) = traindata()

function sigma(x)
    return 1./(1.0+exp.(-x))
end

function leakyReLU(x, alpha=0.1)
    return max.(alpha,x)
end

N = 60000 # number of training points
D = 784 # dimension of each x vector
H1 = 500
H2 = 250
H3 = 100
H4 = 30
batchSize = 10000
batchNum = div(N,batchSize)

xtrain=trainX[:,1:N]./255

# randomly initiate weight and bias
# 784 * 500
W1 = param(randn(H1,D)./sqrt(D))
w1 = param(randn(H1,1)./sqrt(H1))
# 500 * 250
W2 = param(randn(H2,H1)./sqrt(H1))
w2 = param(randn(H2,1)./sqrt(H2))
# 250 * 100
W3 = param(randn(H3,H2)./sqrt(H2))
w3 = param(randn(H3,1)./sqrt(H3))
# 100 * 30
W4 = param(randn(H4,H3)./sqrt(H3))
w4 = param(randn(H4,1)./sqrt(H4))
# 30 * 100
U1 = param(randn(H3,H4)./sqrt(H4))
u1 = param(randn(H3,1)./sqrt(H3))
# 100 * 250
U2 = param(randn(H2,H3)./sqrt(H3))
u2 = param(randn(H2,1)./sqrt(H2))
# 250 * 500
U3 = param(randn(H1,H2)./sqrt(H2))
u3 = param(randn(H1,1)./sqrt(H1))
# 500 * 784
U4 = param(randn(D,H1)./sqrt(H1))
u4 = param(randn(D,1)./sqrt(D))


vW1=zeros(size(W1))
vw1=zeros(size(w1))
vW2=zeros(size(W2))
vw2=zeros(size(w2))
vW3=zeros(size(W3))
vw3=zeros(size(w3))
vW4=zeros(size(W4))
vw4=zeros(size(w4))

vU1=zeros(size(U1))
vu1=zeros(size(u1))
vU2=zeros(size(U2))
vu2=zeros(size(u2))
vU3=zeros(size(U3))
vu3=zeros(size(u3))
vU4=zeros(size(U4))
vu4=zeros(size(u4))

# using CuArrays for GPU support
# w, b, x, y = cu.((w, b, x, y))

encode(x)= leakyReLU(W4*leakyReLU(W3*leakyReLU(W2*leakyReLU(W1*x .+ w1).+w2).+w3).+w4)
decode(h)= sigma(U4*leakyReLU(U3*leakyReLU(U2*leakyReLU(U1*h .+ u1).+u2).+u3).+u4)

#loss(x) = sqrt(0.1+sum((x - decode(encode(x))).^2))/(N*D)
#loss(x) = sum((x - decode(encode(x))).^2)/(N*D)
loss(x) = begin; y=decode(encode(x)); return -sum( x.*log.(y)+(1-x).*log.(1-y))/(N*D); end

# compute minibatch loss
function E(bn)
  x_batch = xtrain[:,(bn-1)*batchSize+1: bn*batchSize]
  loss(x_batch)
end

# momentum
function momentum!(ps, vs, mu=0.9, eta = 5.5)
    for bn = 1 : batchNum
        back!(E(bn))
        @show E(bn)
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
    for i = 1:20
        momentum!((W1, w1, U1, u1, W2, w2, U2, u2, W3, w3, U3, u3, W4, w4, U4, u4),(vW1,vw1,vU1,vu1,vW2,vw2,vU2,vu2,vW3,vw3,vU3,vu3,vW4,vw4,vU4,vu4))
        #@show loss(xtrain)
    end
end

update!()
xrecon=decode(encode(xtrain)).data
pcolormesh(reshape(xrecon[:,1],28,28),cmap="gray")
