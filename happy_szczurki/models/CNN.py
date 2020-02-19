import torch
import torch.nn as nn


def calculate_output_size(input_size, layer):
    data_size = (1, ) + input_size  # 1 represents batch size
    out = layer(torch.zeros(data_size))
    
    return out.size()


class ConvNet(nn.Module):
    def __init__(self, input_shape, layer_params, hidden_dim, activation_fn=nn.ReLU, dropout_p=0.1):
        super().__init__()

        layers = []

        in_channels = input_shape[0]
        for layer in layer_params:
            layers += [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=layer["output_size"],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"]
                ),
                activation_fn(),
                nn.MaxPool2d(
                    kernel_size=layer["max_pool_kernel"],
                    stride=layer["max_pool_stride"]
                )
            ]
            in_channels = layer['output_size']

        layers.append(nn.Flatten())

        seq = nn.Sequential(*layers)

        conv_output_size = calculate_output_size(input_shape, seq)[1]
        print('Calculated output size:', conv_output_size)

        linear_layers_sizes = [conv_output_size] + hidden_dim
        for in_size, out_size in zip(linear_layers_sizes[0:], linear_layers_sizes[1:]):
            layers.extend([
                nn.Dropout(p=dropout_p),
                nn.Linear(in_size, out_size)
            ])

        layers.append(nn.Softmax(dim=1))

        self.layers = nn.Sequential(*layers)

        
    def forward(self, x):
        return self.layers(x)
    
    @property
    def n_param(self):
        return sum(p.numel() for p in self.parameters())