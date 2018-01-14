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
trainX, trainY = traindata()

function sigma(x)
    return 1./(1.0+exp.(-x))
end

function relu(x)
    return max(0,x)
end

function leakyrelu(x)
    return max(alpha,x)
end


N = 50 # number of training points
D = 784 # dimension of each x vector
H1 = 500
H2 = 250
H3 = 100
H4 = 30
numUnits = {D,H1,H2,H3,H4,H3,H2,H1,D}
numLayers = 8
alpha = 0.1 # constant for leaky relu
xtrain=trainX[:,1:N]./255


# randomly initiate weight and bias
for i = 1:numLayers/2
    en_weights[i] = param(randn(numsUnits[i], numsUnits[i+1])./sqrt(numsUnits[i+1]))
    en_bias[i] = param(randn(numUnits[i], 1))./sqrt(numsUnits[i+1])
    en_vW[i] = zeros(size(weights[i]))
    en_vb[i] = zeros(size(bias[i]))
end

for i = numLayers/2+1:numLayers
    de_weights[i] = param(randn(numsUnits[i], numsUnits[i+1])./sqrt(numsUnits[i+1]))
    de_bias[i] = param(randn(numUnits[i], 1))./sqrt(numsUnits[i+1])
    de_vW[i] = zeros(size(weights[i]))
    de_vb[i] = zeros(size(bias[i]))
end

encode(x)= sigma(W2*sigma(W1*x .+ w1).+w2)
decode(h)= sigma(U2*sigma(U1*h .+ u1).+u2)

function model(x)
    for i = 1:numberLayers-1
        x = leakyrelu(weights[i]*x+bias[i])
    end
    sigma(x)
end




#=
# 784 * 250
W1 = param(randn(Q,D)./sqrt(D))
w1 = param(randn(Q,1)./sqrt(D))
# 250 * 20
W2 = param(randn(H,Q)./sqrt(Q))
w2 = param(randn(H,1)./sqrt(Q))
# 20 * 250
U1 = param(randn(Q,H)./sqrt(H))
u1 = param(randn(Q,1)./sqrt(H))
# 250 * 784
U2 = param(randn(D,Q)./sqrt(Q))
u2 = param(randn(D,1)./sqrt(Q))

vW1=zeros(size(W1))
vw1=zeros(size(w1))
vW2=zeros(size(W2))
vw2=zeros(size(w2))

vU1=zeros(size(U1))
vu1=zeros(size(u1))
vU2=zeros(size(U2))
vu2=zeros(size(u2))
=#

# using CuArrays for GPU support
# w, b, x, y = cu.((w, b, x, y))



#loss(x) = sqrt(0.1+sum((x - decode(encode(x))).^2))/(N*D)
#loss(x) = sum((x - decode(encode(x))).^2)/(N*D)
loss(x) = begin; y=decode(encode(x)); return -sum( x.*log(y)+(1-x).*log(1-y))/(N*D); end

# Gradient Descent
function update!(ps, eta = 1.5)
    for pars in ps
        pars.data .-= pars.grad .* eta  # apply update
        pars.grad .= 0 # clear the gradient, as gradient accumulates in AD?
    end
end

# momentum
function momentum!(ps, vs, mu=0.9, eta = 5.5)
    back!(loss(xtrain))
    for count=1:length(ps)
        copy!(vs[count],  mu*vs[count] + (1-mu)*ps[count].grad)
        copy!(ps[count].data, ps[count].data -eta*vs[count])
    end
    for pars in ps
        pars.grad .= 0 # clear the gradient
    end
end

for i = 1:20
    #update!((W1, w1, U1, u1))
    momentum!((W1, w1, U1, u1,W2, w2, U2, u2),(vW1,vw1,vU1,vu1,vW2,vw2,vU2,vu2))
    @show loss(xtrain)
end

xrecon=decode(encode(xtrain)).data
pcolormesh(reshape(xrecon[:,1],28,28),cmap="gray")
