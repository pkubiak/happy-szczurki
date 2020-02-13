from happy_szczurki.models.CNN import ConvNet
from skorch import NeuralNetClassifier


net = NeuralNetClassifier(
    ConvNet,
    max_epochs=10,
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)

print(net)
net.set_params(module__input_shape=(1, 257,257), module__layer_params=[
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
    ], module__hidden_dim=[39,2,3])
print(net.get_params())




