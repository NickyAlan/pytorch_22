'''
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
from : https://github.com/aladdinpersson/


Information about architecture config:
Tuple is structured --> (kernel_size, filters, stride, padding) 
"M" is  maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
'''

import torch
from torch import nn 

config = [
    (7, 64, 2, 3),
    'M',
    (3, 192, 1, 1),
    'M',
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    'M',
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'M',
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module) :
    def __init__(self, in_channels, out_channels, **kwargs) :
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.leakyRelu = nn.LeakyReLU(0.1)

    def forward(self, x) :
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.leakyRelu(x)
        return x

class YoloV1(nn.Module) :
    def __init__(self, in_channels=3, **kwargs) :
        super(YoloV1, self).__init__()
        self.config = config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.config)
        self.dense = self._create_dense(**kwargs)
        self.flatten = nn.Flatten()

    def _create_conv_layers(self, config) :
        layers = []
        in_channels = self.in_channels

        for layer in config :
            if isinstance(layer, tuple) :
                layers += [ 
                    CNNBlock(in_channels, out_channels=layer[1], kernel_size=layer[0], stride=layer[2], padding=layer[-1])
                    ]
                in_channels = layer[1] # update for input next layer
            
            elif isinstance(layer, str) : # 'M' --> maxpool2D
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
            
            elif isinstance(layer, list) :
                conv1 = layer[0]
                conv2 = layer[1]
                repeats = layer[-1]

                for _ in range(repeats) :
                    layers += [ 
                        CNNBlock(in_channels, out_channels=conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[-1])
                    ]
                    layers += [ 
                        CNNBlock(in_channels=conv1[1], out_channels=conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])
                    ]
                    
                    in_channels = conv2[1] # update for input next layer

        return nn.Sequential(*layers)

    def _create_dense(self, split_size, num_boxes, num_classes) :
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential( 
            nn.Flatten(),
            nn.Linear(1024*S*S, 496),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S*S*(C+B*5))
        )

    def forward(self, x) :
        x = self.darknet(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == '__main__' :
    model = YoloV1(split_size=7, num_boxes=2, num_classes=20)
    x = torch.randn((5, 3, 448, 448)) # batch_size, channel, w, h
    print(model(x).shape) # [5, 1470] --> S*S*(C+B*5) = 7*7*(20+2*5) = 1470