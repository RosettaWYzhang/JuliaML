# use mini batch to perform gradient descent
# TODO: Viasualize error over iterations

using Flux.Tracker
using Plots

N=1000 # number of training points
D=3 # dimension of each x vector
batchSize = 50
batchNum = convert(Int, N/batchSize)
iterations = 10
#η = 0.5
#k = 0.1
#λ = 0.9 #for momentum
#Vdw = 0
#Vdb = 0


# fake dataset
x=randn(D,N)
w_david=randn(1,D)
b_david=randn()
# assume we try to predict a scalar output
y=zeros(1,N)
for n=1:N
    y[n] = sum(w_david * x[:,n] .+ b_david)
end

#visualise the dataset
plotly() # Choose the Plotly.jl backend for web interactivity
plot(x,y,seriestype=:scatter,title="dataset")


w = param(randn(1,D))
b = param(randn())

w = param(randn(1,D))
b = param(randn())

function E(bn)
  x_batch = x[:,(bn-1)*batchSize+1: bn*batchSize]
  y_batch = y[(bn-1)*batchSize+1: bn*batchSize]
  meansquareloss(predict(x_batch),y_batch)
end

# Minibatch gradient descent
function update!(w, b, i; k = 0.1, Vdw = 0, Vdb = 0, eta = 0.5, λ = 0.9)
    Vdw = w.grad * (1-λ) + Vdw * λ
    Vdb = b.grad * (1-λ) + Vdb * λ
    w.data .-= Vdw .* eta
    b.data .-= Vdb .* eta
    w.grad .= 0
    b.grad .= 0;
    return Vdw, Vdb
end

x = 1:10; y = rand(10,2) # 2 columns means two lines
plot(x,y,title="Two Lines",label=["Line 1" "Line 2"],lw=3)


for i = 1:iterations
  for bn = 1:batchNum
    # minibatch gradient descent
    back!(E(bn))
    update!(w,b,i,k=0)
    e1 = E(bn)
    #push!(plt,i,E(bn))
    #@show E(bn)
    # with learning rate annealing
    back!(E(bn))
    update!(w,b,i)
    e2 = E(bn)
    #push!(plt,2,i,E(bn))
    #@show E(bn)
    # with learning rate annealing and momentum
    back!(E(bn))
    Vdw, Vdb = update!(w,b,i,Vdw=Vdw,Vdb=Vdb)
    e3 = E(bn)
    e = [x,y,z]
    push!(plt,t,e3)
    #@show E(bn)

  end
end
