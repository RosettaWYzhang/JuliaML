using Flux.Tracker

# random dataset
x, y = rand(5), rand(2)

# randomly initiate weight and bias
W = param(rand(2, 5))
b = param(rand(2))

predict(x) = W*x .+ b
meansquarederror(ŷ, y) = sum((ŷ .- y).^2)/size(y, 2)
loss(x, y) = meansquarederror(predict(x), y)

function update!(ps, η = .1)
  for w in ps
    w.data .-= w.grad .* η
    w.grad .= 0
  end
end


for i = 1:5
  back!(loss(x, y))
  update!((W, b))
  @show loss(x, y)
end
