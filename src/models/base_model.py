import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, original_img_size=[72, 48]):
        super(BaseModel, self).__init__()

        self.height = original_img_size[0]
        self.width = original_img_size[1]

        self.encoder_1 = nn.Sequential(
            nn.Linear(self.height*self.width, 1024),  # 72x48, 1024
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )

        self.middle_layer_1 = self.make_layers(512, num_repeat=100)

        self.encoder_2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )

        self.decoder_1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )

        self.decoder_2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, self.height*self.width),
        )

    def make_layers(self, value, num_repeat):
        layers = []
        for _ in range(num_repeat):
            layers.append(nn.Linear(value, value))
            layers.append(nn.BatchNorm1d(value))
            layers.append(nn.LeakyReLU(inplace=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.height*self.width)
        # print(x)
        # print(type(x))
        # print(x.shape)
        x = self.encoder_1(x)
        x = self.middle_layer_1(x)
        x = self.encoder_2(x)
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = x.view(-1, 1, self.height, self.width)
        return x
