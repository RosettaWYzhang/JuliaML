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

function leakyReLU(x,alpha=0.1)
    return max(alpha,x)
end


N = 50 # number of training points
D = 784 # dimension of each x vector
numUnits = [784,500,250,100,30,100,250,500,784]
numLayers = 8
xtrain=trainX[:,1:N]./255

# randomly initiate weight and bias
for i = 1:numLayers
    weights[i] = param(randn(numsUnits[i], numsUnits[i+1])./sqrt(numsUnits[i+1]))
    bias[i] = param(randn(numUnits[i], 1))./sqrt(numsUnits[i+1])
    vW[i] = zeros(size(weights[i]))
    vb[i] = zeros(size(bias[i]))
end

chain(x, i) = leakyReLU(weights[i]*x .+ bias[i])

function model(x)
    for i = 1:numberLayers-1
        x = chain(x,i)
    end
    return sigma(x*weights[numLayers] .+ bias[numLayers])
end

loss(x) = begin; y=model(x); return -sum( x.*log(y)+(1-x).*log(1-y))/(N*D); end

# update with momentum
function momentum!(weights,bias,vWeights,vbias, mu=0.9, eta = 5.5)
    back!(loss(xtrain))
    for count=1:length(ps)
        copy!(vWeights[count],  mu*vWeights[count] + (1-mu)*weights[count].grad)
        copy!(vBias[count],  mu*vBias[count] + (1-mu)*bias[count].grad)
        copy!(weights[count].data, weights[count].data -eta*vWeights[count])
        copy!(bias[count].data, bias[count].data -eta*vBias[count])
    end
    for pars in ps,bias,vWeights
        pars.grad .= 0 # clear the gradient
    end
end

for i = 1:20
    momentum!(weights,bias,vweights,vbias)
    @show loss(xtrain)
end

xrecon = model(xtrain).data
pcolormesh(reshape(xrecon[:,1],28,28),cmap="gray")
