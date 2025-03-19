### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ e8a55555-b17e-4308-82f3-5099640b3604
import Pkg

# ╔═╡ de0e8a9d-4bea-4f14-b0eb-2621df143bca
Pkg.activate("..")

# ╔═╡ 50fad57e-5a08-43f4-8205-be73009507ce
using Flux

# ╔═╡ 7b853172-e888-40ac-bc47-38d16952f4d2
using ProgressMeter

# ╔═╡ ddde2b2c-7dde-480f-bb2c-62ba33ab0e95
using Statistics

# ╔═╡ a42b2586-346a-41f7-9105-618800b050dc
using Plots 

# ╔═╡ 6d9f28b4-04ca-11f0-1251-3d693c14e1fd
md"""
# Machine learning with Julia and Flux
## Tutorial 2
"""

# ╔═╡ 3f49c169-47dd-4c30-a25f-ff85ae4b885f
md"""
### A Neural Network in One Minute
###### This is the official tutorial of Flux.jl !
"""

# ╔═╡ e2c09819-0a1c-4475-99ce-2aec3698a2a2
md"""
##### Generate some data for the XOR problem
"""

# ╔═╡ d79be718-eb52-478b-960a-0c8941116d6b
# Random pairs of numbers
noisy = rand(Float32, 2, 1000)

# ╔═╡ a633ebf0-874c-42ad-9d2b-459a81d15646
# Xor Answer
truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)] 

# ╔═╡ dc452b60-9344-4e5b-a4a7-46c3ebefc473
md"""
##### Define our model, a multi-layer perceptron with one hidden layer
"""

# ╔═╡ 6a56265e-74ec-4c85-ab76-3708278d9296
model = Chain(
    Dense(2 => 3, tanh),      # activation function inside layer
    BatchNorm(3),
    Dense(3 => 2))

# ╔═╡ acb1a691-8340-4758-ac77-7895399f63a8
md"""
##### Outputs before training
"""

# ╔═╡ 05fe2a99-3fda-4ab3-af53-e2589a9a47b2
out1 = model(noisy)

# ╔═╡ 1a505b26-bd26-4fc5-b3d1-063b3a3a7e37
probs1 = softmax(out1)

# ╔═╡ 3e7f5ddc-016d-44a8-999f-61d85d3550fb
md"""
##### One-hot encoding
"""

# ╔═╡ ed9a29a7-c902-448c-b5ab-5231ea3a943d
target = Flux.onehotbatch(truth, [true, false])

# ╔═╡ 06cc77be-0a55-4f87-9d5e-5cd4b9317ea9
md"""
##### Use Batches of 64 samples
"""

# ╔═╡ d806734c-8822-4f9e-8afb-5f35d64fa5b5
loader = Flux.DataLoader((noisy, target), batchsize=64, shuffle=true)

# ╔═╡ 87efb9c5-5bbe-451a-b9e5-14b763f49b8b
opt_state = Flux.setup(Flux.Adam(0.01), model)

# ╔═╡ ed7ab9ad-ad37-4091-9a17-e6c50f41eca7
losses = []

# ╔═╡ f45643de-3e23-4f84-8dd3-319bb99ec721
md"""
##### Training loop, using the whole data set 1000 times:
"""

# ╔═╡ 81327f9e-98af-4de9-9d95-8fdaaaadb084
@showprogress for epoch in 1:1_000
    for xy_cpu in loader
        x, y = xy_cpu
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.logitcrossentropy(y_hat, y)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end

# ╔═╡ 17905753-1cdc-484c-bf22-fc1b5df199b1
opt_state

# ╔═╡ 1796f3e1-3253-4678-8d48-cbf0959a0121
out2 = model(noisy)  

# ╔═╡ 9993ca9e-45d3-449f-8d1d-ed73461f5b47
probs2 = softmax(out2)

# ╔═╡ 8479d474-bd97-4056-a7e6-6fdd2bf8c0d5
mean((probs2[1,:] .> 0.5) .== truth)

# ╔═╡ 41daaa18-9dd9-4db3-ab85-8dd99d98b221
begin
	p_true = scatter(noisy[1,:], noisy[2,:], zcolor=truth, title="True classification", legend=false, aspect_ratio = :equal)
	p_raw =  scatter(noisy[1,:], noisy[2,:], zcolor=probs1[1,:], title="Untrained network", label="", clims=(0,1), aspect_ratio = :equal)
	p_done = scatter(noisy[1,:], noisy[2,:], zcolor=probs2[1,:], title="Trained network", legend=false, aspect_ratio = :equal)
	
	plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330))
end

# ╔═╡ f6355ee7-c69e-4136-bf7c-3ecb26c6c088
begin
	plot(losses; xaxis=(:log10, "iteration"),
	    yaxis="loss", label="per batch")
	n = length(loader)
	plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)),
	    label="epoch mean", dpi=200)
end

# ╔═╡ cae8490d-51eb-498d-bfd5-6dc27ce7ba21
md"""
#### Note:
	Instead of calling gradient and update! separately, there is a convenience function train!.
"""

# ╔═╡ a1e6e092-9e20-4e90-bae8-373ae4d080f0
for epoch in 1:1_000
    Flux.train!(model, loader, opt_state) do m, x, y
        y_hat = m(x)
        Flux.logitcrossentropy(y_hat, y)
    end
end

# ╔═╡ Cell order:
# ╠═6d9f28b4-04ca-11f0-1251-3d693c14e1fd
# ╠═e8a55555-b17e-4308-82f3-5099640b3604
# ╠═de0e8a9d-4bea-4f14-b0eb-2621df143bca
# ╠═50fad57e-5a08-43f4-8205-be73009507ce
# ╠═7b853172-e888-40ac-bc47-38d16952f4d2
# ╠═ddde2b2c-7dde-480f-bb2c-62ba33ab0e95
# ╟─3f49c169-47dd-4c30-a25f-ff85ae4b885f
# ╟─e2c09819-0a1c-4475-99ce-2aec3698a2a2
# ╠═d79be718-eb52-478b-960a-0c8941116d6b
# ╠═a633ebf0-874c-42ad-9d2b-459a81d15646
# ╟─dc452b60-9344-4e5b-a4a7-46c3ebefc473
# ╠═6a56265e-74ec-4c85-ab76-3708278d9296
# ╟─acb1a691-8340-4758-ac77-7895399f63a8
# ╠═05fe2a99-3fda-4ab3-af53-e2589a9a47b2
# ╠═1a505b26-bd26-4fc5-b3d1-063b3a3a7e37
# ╠═3e7f5ddc-016d-44a8-999f-61d85d3550fb
# ╠═ed9a29a7-c902-448c-b5ab-5231ea3a943d
# ╟─06cc77be-0a55-4f87-9d5e-5cd4b9317ea9
# ╠═d806734c-8822-4f9e-8afb-5f35d64fa5b5
# ╠═87efb9c5-5bbe-451a-b9e5-14b763f49b8b
# ╠═ed7ab9ad-ad37-4091-9a17-e6c50f41eca7
# ╟─f45643de-3e23-4f84-8dd3-319bb99ec721
# ╠═81327f9e-98af-4de9-9d95-8fdaaaadb084
# ╠═17905753-1cdc-484c-bf22-fc1b5df199b1
# ╠═1796f3e1-3253-4678-8d48-cbf0959a0121
# ╠═9993ca9e-45d3-449f-8d1d-ed73461f5b47
# ╠═8479d474-bd97-4056-a7e6-6fdd2bf8c0d5
# ╠═a42b2586-346a-41f7-9105-618800b050dc
# ╟─41daaa18-9dd9-4db3-ab85-8dd99d98b221
# ╠═f6355ee7-c69e-4136-bf7c-3ecb26c6c088
# ╟─cae8490d-51eb-498d-bfd5-6dc27ce7ba21
# ╠═a1e6e092-9e20-4e90-bae8-373ae4d080f0
