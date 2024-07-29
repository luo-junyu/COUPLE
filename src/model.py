import torch
import torch.nn as nn

class DDPA(nn.Module):
    def __init__(self, args, num_class, dim1):
        super(DDPA, self).__init__()

        self.n_class = num_class

        self.encoder = nn.Sequential(
            nn.Linear(dim1, dim1),
            nn.BatchNorm1d(dim1),
            nn.ReLU(inplace=False),
        )

        self.projector = nn.Sequential(
            nn.Linear(dim1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, args.nbit),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(dim1, num_class)
        )


    def forward(self, source, target):
        source_f1 = self.encoder(source)
        source_h = self.projector(source_f1)
        source_clf = self.classifier(source_f1)

        target_f1 = self.encoder(target)
        target_h = self.projector(target_f1)
        target_clf = self.classifier(target_f1)

        return source_f1, source_h, source_clf, target_f1, target_h, target_clf

    def forward_clf(self, x):
        x = self.encoder(x)
        clf = self.classifier(x)
        return clf

    def classify(self, feat):
        return self.classifier(feat)

    def predict(self, x):
        x = self.encoder(x)
        h = self.projector(x)
        return h