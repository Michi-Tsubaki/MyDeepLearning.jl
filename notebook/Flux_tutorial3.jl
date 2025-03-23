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

# ╔═╡ ddde2b2c-7dde-480f-bb2c-62ba33ab0e95
using Statistics

# ╔═╡ f3b901b0-0b77-4ebc-8690-fdf7468bf63a
using MLDatasets

# ╔═╡ 5d4e3481-bac9-4e8b-97df-cafb03fd07c3
using Plots

# ╔═╡ f82b0db0-91ae-48a8-a561-34cb90d78d35
using ImageShow

# ╔═╡ 923b6df2-0693-449e-a7d9-598afa8e3698
using Images

# ╔═╡ 6d9f28b4-04ca-11f0-1251-3d693c14e1fd
md"""
# Machine learning with Julia and Flux
## MNIST
In this notebook shows a simple implementation of MLP using Flux.jl
"""

# ╔═╡ 9951f588-aaa7-4f43-9050-8cffb3e0b11c
md"""
### Load Packages
"""

# ╔═╡ 3f49c169-47dd-4c30-a25f-ff85ae4b885f
md"""
#### Load Data
"""

# ╔═╡ 684e7076-81c9-445a-91d8-88eadca85782
# Training Data
X_train_raw, Y_train_raw = MLDatasets.MNIST(split=:train)[:];

# ╔═╡ 1f9dd749-8434-4a51-93d0-470d70f48fa8
# Size
size(X_train_raw)

# ╔═╡ c8f3886a-85b8-4b52-8ce1-70520fc37950
index = 1;

# ╔═╡ ea88f0c4-9a38-4530-a96e-90f3398754d2
X_train_raw[:,:,index]

# ╔═╡ b20df8d0-45b9-472b-9876-4f0e08fa8f69
Y_train_raw[index]

# ╔═╡ 79f20dd9-c18b-4179-8307-e6fb6f3f5ab3
# Test Data
X_test_raw, Y_test_raw = MLDatasets.MNIST(split=:test)[:];

# ╔═╡ 51e77cbb-4d99-43b9-8b22-1a5439c265a0
# Size
size(X_test_raw)

# ╔═╡ aaea1f73-a9a0-496b-ae06-cbac9715e384
md"""
### Visualise Training Data
"""

# ╔═╡ 6c08d526-1cfc-4b6d-81cf-97fda1176494
convert2image(MNIST(), 1)

# ╔═╡ 8e3e74b6-b274-4067-b728-981806971402
md"""
### Preprocess
"""

# ╔═╡ f49824b9-107b-41e7-b265-96b9385e4566
x_train = Flux.flatten(X_train_raw)

# ╔═╡ 6eb8c5f5-344e-4462-96e0-61ea949f8f3b
x_test = Flux.flatten(X_test_raw);

# ╔═╡ 90a0feb5-70fa-45a8-81c1-230bf4ad9adc
y_train = Flux.onehotbatch(Y_train_raw,0:9)

# ╔═╡ b89110a5-d5b8-4d6d-a2b2-09394fda5e06
y_test = Flux.onehotbatch(Y_test_raw,0:9);

# ╔═╡ fbba3b82-77a7-4226-88ee-a1de867bd42c
md"""
### Model
"""

# ╔═╡ 06821ff6-88df-458c-91d4-241c6229c0c9
model = Chain(
    Dense(28 * 28, 32, relu), # input_dim, output_dim, activate_func
    Dense(32, 10), # input_dim, output_dim, activate_func
    softmax
)

# ╔═╡ 98c818f6-3d1a-4466-a492-0bea214c4e5e
md"""
### Loss Function
"""

# ╔═╡ 3a15cc21-676e-44f8-a4c4-a83f98b71bb7
# Loss function - 1
function loss(model,x, y)
    temp = x
    for layer in model.layers
        temp = layer(temp)
    end
    return Flux.crossentropy(temp, y)
end

# ╔═╡ 92ea71f1-b8d7-4503-9bee-cbfb8b5bd0ba
# Loss function - 2
function loss(x, y)
    return Flux.crossentropy(model(x), y)
end

# ╔═╡ 54a114f7-732c-49bf-a2a8-23350d11306b
md"""
### Parameters
"""

# ╔═╡ e45d61bf-3e09-4708-a0fb-420dce25ed5c
# Parameters
ps = Flux.trainable(model)

# ╔═╡ 1bef8414-9d2a-44eb-a5ab-9cd39b5a3232
learning_rate = 0.01

# ╔═╡ 199aacf8-69b9-46e8-9902-7fb5ced1825b
epochs = 100

# ╔═╡ b2601f08-b380-4759-a323-859a66e12855
opt = Flux.setup(Adam(learning_rate), model)

# ╔═╡ d837f20a-e0fa-447b-a4fc-c89040a211b0
md"""
### Train
"""

# ╔═╡ 0c1d0824-6239-42cc-a0db-8bbd8ad5c942
function train_loop(ps, opt; epochs = 100, learning_rate = 0.01)
	loss_history = []
	@time for epoch in 1:epochs
	    data = Flux.DataLoader((x_train, y_train), batchsize=size(x_train, 2))
		Flux.train!(loss, ps, data, opt)
	    train_loss = loss(x_train, y_train)
	    push!(loss_history, train_loss)
	    println("Epoche = $epoch : Verlust = $train_loss")
	end
	return loss_history
end

# ╔═╡ 3c42077a-7e16-4437-8efc-79232062bffd
train_loop(ps, opt; epochs, learning_rate)

# ╔═╡ ef6dab0e-e4e5-45bf-bb5b-d351ffff1457
md"""
### Test
"""

# ╔═╡ 3b68b8a7-d4bb-4b16-9ad9-d88b8add3f1d
function test_loop(x_test, Y_test_raw)
	y_predicted_raw = model(x_test)
	y_predicted = Flux.onecold(y_predicted_raw).-1
	result = y_predicted .== Y_test_raw
	acc = mean(result)
	return acc, result
end

# ╔═╡ e8fc879c-f6d5-41d6-82da-dd418c8ddd96
accuracy, results = test_loop(x_test, Y_test_raw)

# ╔═╡ e9f03af6-843c-495e-be45-91f719a7ca73
println("正解率は", accuracy*100, "%です．")

# ╔═╡ 6a582a95-6c00-46e7-b726-5b9222a79b7a
results

# ╔═╡ Cell order:
# ╟─6d9f28b4-04ca-11f0-1251-3d693c14e1fd
# ╟─9951f588-aaa7-4f43-9050-8cffb3e0b11c
# ╠═e8a55555-b17e-4308-82f3-5099640b3604
# ╠═de0e8a9d-4bea-4f14-b0eb-2621df143bca
# ╠═50fad57e-5a08-43f4-8205-be73009507ce
# ╠═ddde2b2c-7dde-480f-bb2c-62ba33ab0e95
# ╠═f3b901b0-0b77-4ebc-8690-fdf7468bf63a
# ╠═5d4e3481-bac9-4e8b-97df-cafb03fd07c3
# ╠═f82b0db0-91ae-48a8-a561-34cb90d78d35
# ╠═923b6df2-0693-449e-a7d9-598afa8e3698
# ╟─3f49c169-47dd-4c30-a25f-ff85ae4b885f
# ╠═684e7076-81c9-445a-91d8-88eadca85782
# ╠═1f9dd749-8434-4a51-93d0-470d70f48fa8
# ╠═c8f3886a-85b8-4b52-8ce1-70520fc37950
# ╠═ea88f0c4-9a38-4530-a96e-90f3398754d2
# ╠═b20df8d0-45b9-472b-9876-4f0e08fa8f69
# ╠═79f20dd9-c18b-4179-8307-e6fb6f3f5ab3
# ╠═51e77cbb-4d99-43b9-8b22-1a5439c265a0
# ╟─aaea1f73-a9a0-496b-ae06-cbac9715e384
# ╠═6c08d526-1cfc-4b6d-81cf-97fda1176494
# ╟─8e3e74b6-b274-4067-b728-981806971402
# ╠═f49824b9-107b-41e7-b265-96b9385e4566
# ╠═6eb8c5f5-344e-4462-96e0-61ea949f8f3b
# ╠═90a0feb5-70fa-45a8-81c1-230bf4ad9adc
# ╠═b89110a5-d5b8-4d6d-a2b2-09394fda5e06
# ╠═fbba3b82-77a7-4226-88ee-a1de867bd42c
# ╠═06821ff6-88df-458c-91d4-241c6229c0c9
# ╟─98c818f6-3d1a-4466-a492-0bea214c4e5e
# ╠═3a15cc21-676e-44f8-a4c4-a83f98b71bb7
# ╠═92ea71f1-b8d7-4503-9bee-cbfb8b5bd0ba
# ╟─54a114f7-732c-49bf-a2a8-23350d11306b
# ╠═e45d61bf-3e09-4708-a0fb-420dce25ed5c
# ╠═1bef8414-9d2a-44eb-a5ab-9cd39b5a3232
# ╠═199aacf8-69b9-46e8-9902-7fb5ced1825b
# ╠═b2601f08-b380-4759-a323-859a66e12855
# ╟─d837f20a-e0fa-447b-a4fc-c89040a211b0
# ╠═0c1d0824-6239-42cc-a0db-8bbd8ad5c942
# ╠═3c42077a-7e16-4437-8efc-79232062bffd
# ╟─ef6dab0e-e4e5-45bf-bb5b-d351ffff1457
# ╠═3b68b8a7-d4bb-4b16-9ad9-d88b8add3f1d
# ╠═e8fc879c-f6d5-41d6-82da-dd418c8ddd96
# ╠═e9f03af6-843c-495e-be45-91f719a7ca73
# ╠═6a582a95-6c00-46e7-b726-5b9222a79b7a
