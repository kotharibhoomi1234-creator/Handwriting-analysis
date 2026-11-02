import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from graphology_db import GRAPHOLOGY_DATABASE

class HandwritingGraphologyModel(nn.Module):
    def __init__(self, num_classes=26):
        super(HandwritingGraphologyModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out

def run_pca_reduction(X, n_components=50):
    X_flat = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=min(n_components, X_flat.shape[1], X_flat.shape[0]))
    return pca.fit_transform(X_flat), pca
