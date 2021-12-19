import os

import matplotlib as mpl
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import sklearn
import sklearn.metrics
import torch

mpl.use('Agg')
from matplotlib import pyplot as plt
from pytorch_lightning.core.lightning import LightningModule
from dqo.estimator.others.neo import evaluate as ne
from torch.nn import functional as F

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


class NeoRegressor(LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(89, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch['input'], batch['runtime']
        y_hat = self(x)
        loss = F.smooth_l1_loss(y_hat.view(-1).double(), y)
        self.log('train_loss', loss)

        return loss

    def on_validation_epoch_start(self):
        self.validation_predictions = []
        self.validation_targets = []

    def validation_step(self, batch, batch_idx):
        x, y = batch['input'], batch['runtime']
        y_hat = self(x)
        loss = F.smooth_l1_loss(y_hat.view(-1), y)
        
        self.log('val_loss', loss)

        for i in range(len(y_hat)):
            self.validation_predictions.append(y_hat[i].item())
            self.validation_targets.append(y[i].item())

        return loss

    def on_validation_epoch_end(self, **outputs):
        if len(self.validation_predictions) > 10:
            ne.save_log_results(
                self.validation_predictions,
                self.validation_targets,
                path=self.trainer.logger.log_dir,
                prefix=f'val_results_{self.current_epoch}'
            )
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
