using MLDatasets

train_x, train_y_raw = MLDatasets.MNIST(split=:train)[1:100]