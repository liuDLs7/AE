import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, ipt_size, opt_size):
        super(AutoEncoder, self).__init__()

        self.hidden_size = int(ipt_size / 3)

        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(ipt_size, self.hidden_size),
            nn.ReLU(),
            # nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Linear(self.hidden_size, opt_size),
            nn.ReLU(),
        )

        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(opt_size, self.hidden_size),
            nn.ReLU(),
            # nn.Linear(128, 1024),
            # nn.ReLU(),
            nn.Linear(self.hidden_size, ipt_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
