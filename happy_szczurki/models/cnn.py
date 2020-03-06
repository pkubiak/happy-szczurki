import torch
import torch.nn as nn
from ..utils import calculate_output_sizes


STR_TO_ACTIVATION = {
    'relu': torch.nn.ReLU,
}

STR_TO_POOL = {
    'max': nn.MaxPool2d,
    'avg': nn.AvgPool2d,
}

class ListWrapper(list):
    def __lshift__(self, other):
        self.append(other)


class ConvNet(nn.Module):
    def __init__(self, input_shape, conv_layers, linear_layers, classes):
        super().__init__()

        layers = ListWrapper()

        in_channels = input_shape[0]
        for layer in conv_layers:
            # NOTE: layers order based on: https://www.researchgate.net/figure/Proposed-Modified-ResNet-18-architecture-for-Bangla-HCR-In-the-diagram-conv-stands-for_fig1_323063171
            if layer.get('conv'):
                layers << nn.Conv2d(in_channels=in_channels, **layer['conv'])
                in_channels = layer['conv']['out_channels']

            if layer.get('dropout'):
                layers << nn.Dropout(p=layer['dropout'])

            if layer.get('batch_norm'):
                layers << nn.BatchNorm2d(in_channels, track_running_stats=False)

            if layer.get('activation'):
                activation_fn = STR_TO_ACTIVATION[layer['activation']]
                layers << activation_fn()
            
            if layer.get('pool'):
                params = dict(layer['pool'])
                pool_fn = STR_TO_POOL[params.pop('type')]

                layers << pool_fn(**params)

        layers << nn.Flatten()

        in_features = calculate_output_sizes(input_shape, layers)[-1][0]

        for layer in linear_layers:
            layers << nn.Linear(in_features, layer['out_features'])
            
            if layer.get('dropout'):
                layers << nn.Dropout(p=layer['dropout'])

            if layer.get('activation'):
                activation_fn = STR_TO_ACTIVATION[layer['activation']]
                layers << activation_fn()
            
            in_features = layer['out_features']

        layers << nn.Linear(in_features, classes)
        layers << nn.Softmax(dim=1)

        self.layers = nn.Sequential(*layers)

        
    def forward(self, x):
        return self.layers(x)
    
    @property
    def n_param(self):
        return sum(p.numel() for p in self.parameters())


