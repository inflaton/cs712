import torch
import torch.nn as nn
import torch.nn.functional as F


class JigsawNet(nn.Module):
    def __init__(self, n_classes=50, num_features=2048):
        super(JigsawNet, self).__init__()

        self.num_features = num_features

        self.fc1 = nn.Linear(num_features, 512)
        self.fc2 = nn.Linear(18432, 16384)
        self.fc3 = nn.Linear(16384, 4096)
        self.fc4 = nn.Linear(4096, n_classes)
        self.bn4 = nn.BatchNorm1d(n_classes)  # Batch normalization after fc4
        # self.fc5 = nn.Sequential(
        # nn.Dropout(0.1),
        # nn.Linear(in_features=n_classes, out_features=n_classes),
        # )

    def process_features(self, x):
        res = []
        for i in range(36):
            p = x[i]
            p = self.fc1(p)
            res.append(p)

        p = torch.cat(res, 0)
        return p

    def forward(self, x):
        if x.shape[1] == self.num_features:  # single
            p = self.process_features(x)
            p = p.view(1, -1)
        else:  # batch
            res = []
            for i in x:
                p = self.process_features(i)
                res.append(p)

            p = torch.cat(res, 0).view(x.shape[0], -1)

        x = F.relu(p)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = F.relu(
            self.bn4(self.fc4(x))
        )  # Apply batch normalization after fc4 and before activation

        # x = self.fc5(x)
        x = F.softmax(x, dim=1)

        return x
