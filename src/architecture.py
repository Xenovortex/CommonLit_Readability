from torch import nn


class Net(nn.Module):
    """3-layer Neural Network"""

    def __init__(self, num_features, num_hidden, num_output, use_BN=False, use_Dropout=False):
        super(Net, self).__init__()

        if use_BN and use_Dropout:
            self.model = nn.Sequential(
                nn.Linear(num_features, num_hidden),
                nn.ReLU(True),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(True),
                nn.BatchNorm1d(num_hidden),
                nn.Dropout(p=0.2),
                nn.Linear(num_hidden, num_output),
            )
        elif use_BN:
            self.model = nn.Sequential(
                nn.Linear(num_features, num_hidden),
                nn.ReLU(True),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(True),
                nn.BatchNorm1d(num_hidden),
                nn.Linear(num_hidden, num_output),
            )
        elif use_Dropout:
            self.model = nn.Sequential(
                nn.Linear(num_features, num_hidden),
                nn.ReLU(True),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(True),
                nn.Dropout(p=0.2),
                nn.Linear(num_hidden, num_output),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(num_features, num_hidden),
                nn.ReLU(True),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(True),
                nn.Linear(num_hidden, num_output),
            )

    def forward(self, x):
        return self.model(x)
