from happy_szczurki.models.CNN import ConvNet
# from happy_szczurki.datasets import Dataset

from skorch import NeuralNetClassifier
import numpy as np
import torch


net = NeuralNetClassifier(
    ConvNet,
    max_epochs=20,
    lr=0.1,
    # Shuffle training data on each epoch
    # iterator_train__shuffle=True,
)

input_shape = (1, 31, 31)

net.set_params(module__input_shape=input_shape, module__layer_params=[
    {
        "output_size": 37,
        "kernel_size": 5,
        "stride": 1,
        "padding": 2,
        "max_pool_kernel": 3,
        "max_pool_stride": 2
    },
    {
        "output_size": 3,
        "kernel_size": 5,
        "stride": 1,
        "padding": 2,
        "max_pool_kernel": 3,
        "max_pool_stride": 2
    },
    ], module__hidden_dim=[39,2,10])

N = 100
for i in range(10):
    X = np.zeros((N,)+input_shape)
    ys = []

    for i in range(N):
        y = np.random.randint(2)
        X[i,:] = np.random.normal(y, 0.9)
        ys.append(y)

    # X = np.random.random((N, ) + input_shape)
    # y = np.random.randint(0, 10, size=(N,))
    y = np.array(ys)

    X = torch.from_numpy(X).float()
    net.partial_fit(X, y)




