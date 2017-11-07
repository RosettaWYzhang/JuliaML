using Flux.Tracker
#using Plots
#using CuArrays


# random dataset
x, y = rand(5), rand(2)

# visualise dataset
#=plotly() # Choose the Plotly.jl backend for web interactivity
plot(x,y,seriestype=:scatter,title="linear regression")=#


# randomly initiate weight and bias
W = param(rand(2, 5))
b = param(rand(2))

# using CuArrays for GPU support
# W, b, x, y = cu.((W, b, x, y))

predict(x) = W*x .+ b
meansquarederror(ŷ, y) = sum((ŷ .- y).^2)/size(y, 2)
loss(x, y) = meansquarederror(predict(x), y)

function update!(ps, η = .1)
  for w in ps
    w.data .-= w.grad .* η  # apply update
    w.grad .= 0             # clear the gradient, as gradient accumulates in AD?
  end
end


for i = 1:5
  back!(loss(x, y))
  update!((W, b))
  @show loss(x, y)
end
