from happy_szczurki.models.CNN import ConvNet
from happy_szczurki.datasets import Dataset

from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize

# labels = ['22-kHz', '22-kHz call', 'SH', 'FM', 'RP', 'FL', 'ST', 'CMP', 'IU', 'TR', 'RM', 'None'],

mapping = {
    None: 0,
    '22-kHz': 1,
    'SH': 2,
    'FM': 3,
    'RP': 4,
    'FL': 5,
    'ST': 6,
    'CMP': 7,
    'IU': 8,
    'TR': 9,
    'RM': 10
}

net = NeuralNetClassifier(
    ConvNet,
    max_epochs=5,
    lr=0.1,
    # Shuffle training data on each epoch
    # iterator_train__shuffle=True,
    train_split=CVSplit(5, stratified=False)
)

input_shape = (1, 65, 65)

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
    ], module__hidden_dim=[39,2, 11])

dataset = Dataset('data/converted/ch1-2018-11-20_10-20-34_0000006.wav.npz')

N = 500
for epoch in range(20):
    X, y = dataset.sample(N, balanced=True, x_with_frame=True, use_mapping=mapping)
    X = resize(X, (N,) + input_shape[1:])
    X = X.reshape(-1, *input_shape)

    net.partial_fit(X, y)

# TODO: zapisywanie modelu


# TODO: tesotwanie

# TODO: plotowanie lossów
