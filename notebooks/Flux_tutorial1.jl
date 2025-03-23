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
using Zygote

# ╔═╡ 5cf18b04-5325-46cb-8954-fc84ec5bd8d6
using Random

# ╔═╡ ffca69f6-631d-47fa-b49d-ddaa8e3af918
using JuMP

# ╔═╡ 530c1e34-783a-4d61-86b0-72d729f408f2
using Ipopt

# ╔═╡ 6d9f28b4-04ca-11f0-1251-3d693c14e1fd
md"""
# Machine learning with Julia and Flux
## Tutorial 1
"""

# ╔═╡ 3f49c169-47dd-4c30-a25f-ff85ae4b885f
md"""
### Basic Functions in Flux
#### Gradient
"""

# ╔═╡ dcc30300-7085-4abe-92b2-ffb47a72e7ad
md"""
##### Suppose:

$f(x) = x^2 + 4x + 4$

"""

# ╔═╡ ba550978-19e5-44d5-a309-268165a256f1
f(x) = x^2 + 4x + 4

# ╔═╡ ee939a0d-3691-45ef-bf1c-23e8c40031f7
md"""
##### Suppose:

$f(x,y) = \frac{1}{2}\sum (x-y)^2$

"""

# ╔═╡ e91f263f-7d63-49d3-a18b-c252ec05f0ea
f(x, y) = sum((x .- y).^2)*0.5

# ╔═╡ 3a3d4508-5b85-49f3-8b46-a3c8b7d07d74
df(x) = Zygote.gradient(f, x)[1]

# ╔═╡ b8d2a2d0-a5ed-4d95-9366-582082e232d6
d2f(x) = Zygote.gradient(df, x)[1]

# ╔═╡ f1b3edad-f4b5-4d7d-96d2-7245d13dda8d
println(d2f(-2.0))

# ╔═╡ f34b36af-5f59-4f72-802d-6c634da3942d
println(df(-2.0))

# ╔═╡ 54c032ac-db14-443b-a193-8a0e16b8cb37
println(f(-2.0))

# ╔═╡ 40f263c2-ce1a-4768-ba1f-c3dcd96e210a
f_dot(x) = Flux.gradient(f, x)[1]

# ╔═╡ 1b0e852c-1a35-4315-b685-f690a0b25347
println(f_dot(-2.0))

# ╔═╡ b282592c-534d-4f40-a737-10e3aded338e
println(Flux.gradient(f, [2, 1, 4, 3, 5, 6, 7], [2, 0, 3, 6, 3, 7, 4]))

# ╔═╡ 968d1387-4725-4945-83a7-19b0d05fa644
x = [2, 1, 4, 3, 5, 6, 7]

# ╔═╡ 3d85ae7a-78ea-46fb-b5f8-37cb906fed85
y = [2, 0, 3, 6, 3, 7, 4]

# ╔═╡ 40134b20-a6e1-4c29-9e00-2599cbda6100
gs = Zygote.gradient(Params([x, y])) do
         f(x, y)
end

# ╔═╡ 28b500b3-6b07-413d-b86b-ca72a6b23f8a
gs[x] #∂f/∂x

# ╔═╡ 153a6260-932e-4c5d-99ed-5c52c9d8986f
gs[y] #∂f/∂y

# ╔═╡ 3d516c63-55d3-4484-a46f-f4dcf971c0a9
md"""
### Linear Regression
"""

# ╔═╡ 8a70bf3d-7634-4732-856c-54f8f9aa2c32
Random.seed!(2025)

# ╔═╡ 89c6e41e-5a38-49e7-8897-548550906502
W = rand(2,5)

# ╔═╡ 7b1cc41e-0729-49e1-8041-f3df8ef82ad2
b = rand(2)

# ╔═╡ 6edd512b-d9fe-4c39-aa13-7145c3c3c7cf
model(x) = W*x .+ b

# ╔═╡ 77a6b911-8918-4221-8251-842bbbc7b5a7
function loss(x, y)
	ŷ = model(x)
	return 0.5*sum((y .- ŷ).^2)
end

# ╔═╡ daa73c29-9265-485c-a7cd-e82eb59893ab
data_x = rand(5)

# ╔═╡ 17451973-9e28-4a0f-b1d7-fa1189e447bb
data_y = rand(2)

# ╔═╡ 915cb36d-d184-4c61-9784-9960fa498c8e
loss(data_x, data_y)

# ╔═╡ 872ea5ed-bfc4-48e6-a97c-fb0f945fb011
println("x = $(data_x)")

# ╔═╡ 85f71c03-29b2-4992-9ba8-fbe41ab1770d
println("y = $(data_y)")

# ╔═╡ 6243957b-3673-460b-9c81-40a0c6454c94
println("ŷ = $(model(data_x))")

# ╔═╡ 7a9bfe08-03b2-4fa5-a2ae-5beea8272813
println("loss = $(loss(data_x, data_y))")

# ╔═╡ f7720289-2e8c-4c1b-a4d9-aa984ac03a1c
md"""
#### Review: Determine weight by mathematical optimization (JuMP)
"""

# ╔═╡ cf84ea0c-d79e-4ce4-a2c3-e67fa605b141
LSM = Model(Ipopt.Optimizer)

# ╔═╡ 09a7015e-90a1-4d26-9e8d-99a7706cc3dd
set_silent(LSM)

# ╔═╡ c99753c1-060c-4329-8067-0057c4424a5b
num_data = 10

# ╔═╡ 09ea6550-138a-4c6b-a635-7b45ccf5c351
size_data = 5

# ╔═╡ bb7473f5-169f-4372-bfea-a25e12588282
X = rand(num_data,size_data)

# ╔═╡ 48479cfc-8e3b-4c76-9f38-2d20900cee25
y_data = X*[1,2,3,4,5] + rand(10)

# ╔═╡ 1cd4141d-4317-4d3c-9368-556bf2710b77
@variable(LSM, w[1:size_data])

# ╔═╡ 6e8f3421-d581-4b87-9445-6456b3404f5e
@variable(LSM, res[1:num_data])

# ╔═╡ a45b2b95-b302-44c9-81f3-b9953cddf6f8
@constraint(LSM, [i=1:num_data], res[i] == w'*X[i,:] - y_data[i])

# ╔═╡ 3b5a8462-0df8-4daf-8c4b-e25a34211189
@objective(LSM, Min, sum(res[i]^2 for i=1:num_data))

# ╔═╡ 74211ce0-470f-44e5-a529-5bab32581d43
optimize!(LSM)

# ╔═╡ 25fc1070-00cd-4cfc-a580-8fc7f3a477fc
value.(w)

# ╔═╡ cf5ac021-75f6-4b81-91d0-433d0419559c
md"""
### Gradient Decent Method
"""

# ╔═╡ 0994f0aa-19c1-45a7-8458-267cfcb49c32
println(loss(data_x, data_y))

# ╔═╡ b33baae7-d4b3-4b3b-8f9b-d670eff60be5
params = Flux.Params([W, b])

# ╔═╡ 7008b838-1a84-4135-9b24-63cde91204ff
grad_model = Flux.gradient(params) do
    loss(data_x, data_y)
end

# ╔═╡ 0b36c83a-aad9-4d9c-b3b1-15b6d149898f
W̄ = grad_model[W]

# ╔═╡ fe7d3882-ffa6-4d54-bc86-1a83ff39e479
W .-= 0.1 .* W̄

# ╔═╡ 8462215b-cd65-45e2-8b7e-0c6eac9d2441
println(loss(data_x, data_y)) #確かにLOSSが小さくなっている．

# ╔═╡ 5e174a03-b9e1-48d8-922c-22bce74e0c22
md"""
### Layer
"""

# ╔═╡ d95d7528-bd65-474b-8e44-185f7eb7390c
W1 = rand(3, 5)

# ╔═╡ 4e7ff361-e74c-4313-8ad6-da0f9edfbe5c
b1 = rand(3)

# ╔═╡ 2ce54f50-ad35-496a-99a3-4689ed4b12e9
layer1(x) = W1 * x .+ b1

# ╔═╡ 581d2e09-205d-4ab7-bed5-7be8a11d3af5
W2 = rand(2, 3)

# ╔═╡ b44e08c0-f9aa-4bd9-812c-154c0959c327
b2 = rand(2)

# ╔═╡ 06420814-a346-4e8b-9873-a0b55cb29414
layer2(x) = W2 * x .+ b2

# ╔═╡ 4743c017-189c-4f39-8793-e7a6453b02c2
layer_model(x) = layer2(σ.(layer1(x)))

# ╔═╡ 14e930cc-6a23-4ea0-9cef-ca18d8600d77
println(layer_model(rand(5))) # 順伝播

# ╔═╡ 9932bb7d-8073-4eea-80d5-916f7c0dd05c
# Tips

# ╔═╡ 5eaa7ca8-4532-45a9-94cf-f3587cf930ef
function linear(in, out)  # xを引数とする無名関数 x->
    W = randn(out, in)
    b = randn(out)
    x -> W * x .+ b
end

# ╔═╡ 9d6c8061-7265-4f18-9f31-0450335c6b57
linear1 = linear(5, 3)

# ╔═╡ 34925465-fa9e-45fe-8191-e3d63de93525
linear2 = linear(3, 2)

# ╔═╡ 53d094b5-8793-4f06-a36e-aec7cb8d0429
linear_model(x) = linear2(σ.(linear1(x)))

# ╔═╡ 823c04ed-33e0-4052-89c9-0eaefac12be9
println(linear_model(rand(5)))

# ╔═╡ 881aeb83-d3c7-4190-9a98-a9a6d7676141
println(linear1.W) # パラメタアクセス

# ╔═╡ e1d8823e-6e37-406b-98a4-32309d52bdfc
md"""
### Use "Dense"
"""

# ╔═╡ 58a64fb7-adbb-4bf6-abc1-93e653722f26
dense_model = Chain(
    Dense(10, 5, σ),
    Dense(5, 2),
    softmax) #分類

# ╔═╡ 328834c8-ea30-4ea0-97e8-6a7897726d7c
println(dense_model(rand(Float32, 10))) # 指定しないとFloat64になるので注意

# ╔═╡ Cell order:
# ╟─6d9f28b4-04ca-11f0-1251-3d693c14e1fd
# ╠═e8a55555-b17e-4308-82f3-5099640b3604
# ╠═de0e8a9d-4bea-4f14-b0eb-2621df143bca
# ╠═50fad57e-5a08-43f4-8205-be73009507ce
# ╠═7b853172-e888-40ac-bc47-38d16952f4d2
# ╠═5cf18b04-5325-46cb-8954-fc84ec5bd8d6
# ╟─3f49c169-47dd-4c30-a25f-ff85ae4b885f
# ╟─dcc30300-7085-4abe-92b2-ffb47a72e7ad
# ╠═ba550978-19e5-44d5-a309-268165a256f1
# ╠═3a3d4508-5b85-49f3-8b46-a3c8b7d07d74
# ╠═b8d2a2d0-a5ed-4d95-9366-582082e232d6
# ╠═54c032ac-db14-443b-a193-8a0e16b8cb37
# ╠═f34b36af-5f59-4f72-802d-6c634da3942d
# ╠═f1b3edad-f4b5-4d7d-96d2-7245d13dda8d
# ╠═40f263c2-ce1a-4768-ba1f-c3dcd96e210a
# ╠═1b0e852c-1a35-4315-b685-f690a0b25347
# ╟─ee939a0d-3691-45ef-bf1c-23e8c40031f7
# ╠═e91f263f-7d63-49d3-a18b-c252ec05f0ea
# ╠═b282592c-534d-4f40-a737-10e3aded338e
# ╠═968d1387-4725-4945-83a7-19b0d05fa644
# ╠═3d85ae7a-78ea-46fb-b5f8-37cb906fed85
# ╠═40134b20-a6e1-4c29-9e00-2599cbda6100
# ╠═28b500b3-6b07-413d-b86b-ca72a6b23f8a
# ╠═153a6260-932e-4c5d-99ed-5c52c9d8986f
# ╟─3d516c63-55d3-4484-a46f-f4dcf971c0a9
# ╠═8a70bf3d-7634-4732-856c-54f8f9aa2c32
# ╠═89c6e41e-5a38-49e7-8897-548550906502
# ╠═7b1cc41e-0729-49e1-8041-f3df8ef82ad2
# ╠═6edd512b-d9fe-4c39-aa13-7145c3c3c7cf
# ╠═77a6b911-8918-4221-8251-842bbbc7b5a7
# ╠═daa73c29-9265-485c-a7cd-e82eb59893ab
# ╠═17451973-9e28-4a0f-b1d7-fa1189e447bb
# ╠═915cb36d-d184-4c61-9784-9960fa498c8e
# ╠═872ea5ed-bfc4-48e6-a97c-fb0f945fb011
# ╠═85f71c03-29b2-4992-9ba8-fbe41ab1770d
# ╠═6243957b-3673-460b-9c81-40a0c6454c94
# ╠═7a9bfe08-03b2-4fa5-a2ae-5beea8272813
# ╟─f7720289-2e8c-4c1b-a4d9-aa984ac03a1c
# ╠═ffca69f6-631d-47fa-b49d-ddaa8e3af918
# ╠═530c1e34-783a-4d61-86b0-72d729f408f2
# ╠═cf84ea0c-d79e-4ce4-a2c3-e67fa605b141
# ╠═09a7015e-90a1-4d26-9e8d-99a7706cc3dd
# ╠═c99753c1-060c-4329-8067-0057c4424a5b
# ╠═09ea6550-138a-4c6b-a635-7b45ccf5c351
# ╠═bb7473f5-169f-4372-bfea-a25e12588282
# ╠═48479cfc-8e3b-4c76-9f38-2d20900cee25
# ╠═1cd4141d-4317-4d3c-9368-556bf2710b77
# ╠═6e8f3421-d581-4b87-9445-6456b3404f5e
# ╠═a45b2b95-b302-44c9-81f3-b9953cddf6f8
# ╠═3b5a8462-0df8-4daf-8c4b-e25a34211189
# ╠═74211ce0-470f-44e5-a529-5bab32581d43
# ╠═25fc1070-00cd-4cfc-a580-8fc7f3a477fc
# ╟─cf5ac021-75f6-4b81-91d0-433d0419559c
# ╠═0994f0aa-19c1-45a7-8458-267cfcb49c32
# ╠═b33baae7-d4b3-4b3b-8f9b-d670eff60be5
# ╠═7008b838-1a84-4135-9b24-63cde91204ff
# ╠═0b36c83a-aad9-4d9c-b3b1-15b6d149898f
# ╠═fe7d3882-ffa6-4d54-bc86-1a83ff39e479
# ╠═8462215b-cd65-45e2-8b7e-0c6eac9d2441
# ╟─5e174a03-b9e1-48d8-922c-22bce74e0c22
# ╠═d95d7528-bd65-474b-8e44-185f7eb7390c
# ╠═4e7ff361-e74c-4313-8ad6-da0f9edfbe5c
# ╠═2ce54f50-ad35-496a-99a3-4689ed4b12e9
# ╠═581d2e09-205d-4ab7-bed5-7be8a11d3af5
# ╠═b44e08c0-f9aa-4bd9-812c-154c0959c327
# ╠═06420814-a346-4e8b-9873-a0b55cb29414
# ╠═4743c017-189c-4f39-8793-e7a6453b02c2
# ╠═14e930cc-6a23-4ea0-9cef-ca18d8600d77
# ╠═9932bb7d-8073-4eea-80d5-916f7c0dd05c
# ╠═5eaa7ca8-4532-45a9-94cf-f3587cf930ef
# ╠═9d6c8061-7265-4f18-9f31-0450335c6b57
# ╠═34925465-fa9e-45fe-8191-e3d63de93525
# ╠═53d094b5-8793-4f06-a36e-aec7cb8d0429
# ╠═823c04ed-33e0-4052-89c9-0eaefac12be9
# ╠═881aeb83-d3c7-4190-9a98-a9a6d7676141
# ╟─e1d8823e-6e37-406b-98a4-32309d52bdfc
# ╠═58a64fb7-adbb-4bf6-abc1-93e653722f26
# ╠═328834c8-ea30-4ea0-97e8-6a7897726d7c
