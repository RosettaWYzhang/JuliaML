# use mini batch to perform gradient descent
# TODO: find a proper constant k for annealing
# TODO: implement momentum

using Flux.Tracker
#using Plots

N=1000 # number of training points
D=3 # dimension of each x vector
batchSize = 10
batchNum = convert(Int, N/batchSize)
iterations = 100
learning_rate = 0.1
momentum = 0.5
k = 1;

# fake dataset
x=randn(D,N)
w_david=randn(1,D)
b_david=randn()
# assume we try to predict a scalar output
y=zeros(1,N)
for n=1:N
    y[n] = sum(w_david * x .+ b_david)
end

#visualise the dataset
#plotly() # Choose the Plotly.jl backend for web interactivity
#plot(x,y,seriestype=:scatter,title="David's dataset")


# David gives Wanyue only x,y and Wanyue has to try to find w_david and b_david
# randomly initiate weight and bias
w = param(randn(1,D))
b = param(randn())

predict(x) = w*x .+ b
meansquareloss(yhat, y_batch) = sum((yhat - y_batch').^2)/N


function E(bn)
  x_batch = x[:,(bn-1)*batchSize+1: bn*batchSize]
  y_batch = y[(bn-1)*batchSize+1: bn*batchSize]
  meansquareloss(predict(x_batch),y_batch)
end


# Gradient Descent with learning_rate annealing
function update!(ps, i, eta = learning_rate, m = momentum)
  for pars in ps
    #eta_annealing = eta/(i+k)
    eta_annealing = eta;
    pars.data .-= pars.grad .* eta_annealing
    pars.grad .= 0
  end
end


for i = 1:iterations
  for bn = 1:batchNum
    back!(E(bn))
    update!((w, b),i)
    @show E(bn)
  end
end
