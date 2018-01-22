using CuArrays
using Flux.Tracker


N=10 # number of training points
D=3 # dimension of each x vector
# assume we try to predict a scalar output

sigma(x) = 1./(1.0+exp.(-x))
# let's make a fake dataset:
x=randn(D,N)
w_david=randn(D,1)
b_david=randn()
y=zeros(1,N)
for n=1:N
    y[n]=sigma(sum(w_david'*x[:,n])+b_david)
end

# David gives Wanyue only x,y and Wanyue has to try to find w_david and b_david

# randomly initiate weight and bias
w = param(randn(1,D))
b = param([0.])

# using CuArrays for GPU support
w, b, x, y = cu.((w, b, x, y))

predict(x) = sigma(w*x .+ b)
meansquareloss(yhat, y) = sum((yhat - y).^2)/N
#E(x, y) = meansquareloss(predict(x), y)
E2() = meansquareloss(predict(x), y)

# Gradient Descent
function update!(ps, eta = .1)
  for pars in ps
    pars.data .-= pars.grad .* eta  # apply update
    pars.grad .= 0 # clear the gradient, as gradient accumulates in AD?
  end
end

for i = 1:50
  #back!(E(x, y))
    back!(E2())
  update!((w, b))
 # @show E2(x, y)
 @show E2()
end

end

function sigma(x)
return 1./(1.0+exp.(-x))

end
