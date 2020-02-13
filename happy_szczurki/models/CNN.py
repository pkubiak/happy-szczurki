import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, layer_params, hidden_dim):
        super(ConvNet, self).__init__()
        layer_amount = len(layer_params)
        layers = []
        fcs = []
        
        input_size = [1]
        for i in range(layer_amount - 1):
            input_size += [layer_params[i].get("output_size")]

        for i in range(layer_amount):
            layers += [
                nn.Conv2d(input_size[i], layer_params[i].get("output_size"), layer_params[i].get("kernel_size"),
                      layer_params[i].get("stride"), layer_params[i].get("padding")),
                nn.ReLU(),
                nn.MaxPool2d(layer_params[i].get("max_pool_kernel"), layer_params[i].get("max_pool_stride"))
            ]
        self.layers = nn.Sequential(*layers)
        self.drop_out = nn.Dropout()
        
        for i in range(len(hidden_dim) - 1):
            fcs += [nn.Linear(hidden_dim[i], hidden_dim[i + 1])]
        self.fcs = nn.Sequential(*fcs)
        
    def forward(self, x):
        out = self.layers(x)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fcs(out)
        return out
    
    @property
    def n_param(self):
        return sum(p.numel() for p in self.parameters())