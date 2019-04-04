import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


def squash(x):
    x_norm = x.norm(dim=-1, keepdim=True) ** 2
    return (x_norm / (1 + x_norm)) * (x / torch.sqrt(x_norm))


class PrimaryCapsule(BaseModel):
    def __init__(self, num_capsules=8, in_channels=256,
                 out_channels=32, kernel_size=9):
        super(PrimaryCapsule, self).__init__()
        # Initialize the convolutional capsules
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=2,
                      padding=0)
            for _ in range(num_capsules)
        ])

    def forward(self, x):
        # Prediction for each capsule
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * 6 * 6, -1)
        return squash(u)


class DigitCap(BaseModel):
    def __init__(self, num_capsules=10, in_size=32*6*6, in_channels=8,
                 out_channels=16, num_iterations=3):
        super(DigitCap, self).__init__()
        self.num_iterations = 3
        self.W = nn.Parameter(
            torch.randn(num_capsules,in_size, in_channels, out_channels)
        )

    def forward(self, x):
        u_ji = x[:, None, :, None, :] @ self.W[None, :, :, :, :]
        u_ji = u_ji.squeeze(3)
        b_ij = torch.zeros(u_ji.shape, device=x.device)

        for i in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            v_j = squash((c_ij * u_ji).sum(dim=2, keepdim=True))

            if i < self.num_iterations - 1:
                a_ij = (u_ji * v_j).sum(dim=-1, keepdim=True)
                b_ij = b_ij + a_ij

        return v_j.squeeze(2)


class Decoder(BaseModel):
    def __init__(self, in_channels=16*10):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(in_channels, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)

    def forward(self, x, labels):
        mask = torch.sparse.torch.eye(10).to(x.device)
        x = x * mask[labels, :, None]
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.view(-1, 1, 28, 28)


class CapsNet(BaseModel):
    def __init__(self):
        super(CapsNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 256, kernel_size=9)
        self.primarycaps = PrimaryCapsule()
        self.digitcaps = DigitCap()
        self.decoder = Decoder()

    def forward(self, x, labels=None):
        x = F.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)

        classes = x.norm(p=2, dim=-1)
        classes = F.softmax(classes, dim=-1)
        if labels is not None:
            reconstruction = self.decoder(x, labels.squeeze(1))
        else:
            max_class = classes.argmax(dim=1)
            reconstruction = self.decoder(x, max_class)
        return classes, reconstruction

