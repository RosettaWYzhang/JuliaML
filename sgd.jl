# use mini batch to perform gradient descent
# TODO: Debug momentum
# TODO: Viasualize gradient descent

using Flux.Tracker
using Plots

N=5000 # number of training points
D=3 # dimension of each x vector
batchSize = 50
batchNum = convert(Int, N/batchSize)
iterations = 20
η = 2.0
k = 0.1
λ = 0.1 #for momentum
Vdw = 0
Vdb = 0


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
plotlyjs() # Choose the Plotly.jl backend for web interactivity
plot(x,y,seriestype=:scatter,title="dataset")


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


# Minibatch gradient descent
function update!(w, b, i, Vdw = 0, Vdb = 0, eta = η)
    eta = eta/(1+i*k) # learning rate annealing
    w.data .-= w.grad .* eta * (1-λ) + Vdw * λ
    Vdw = w.grad
    w.grad .= 0
    b.data .-= b.grad .* eta * (1-λ) + Vdb * λ
    Vdb = b.grad
    b.grad .= 0;
    return Vdw, Vdb
end


for i = 1:iterations
  for bn = 1:batchNum
    back!(E(bn))
    Vdw, Vdb = update!(w,b,i,Vdw,Vdb)
    @show E(bn)
  end
end
