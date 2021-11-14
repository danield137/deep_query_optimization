import os

import matplotlib as mpl
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch

mpl.use('Agg')
from matplotlib import pyplot as plt
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F
from dqo.estimator.greq import evaluate as ge


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)



class GREQRegressor(LightningModule):
    """
    Generic Relational Query Classifier
    """

    def __init__(self, hidden_size=125, num_layers=2, max_seq=125, bidirectional=True):
        super().__init__()
        self.max_seq = max_seq
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.fc_projection = torch.nn.Linear(14, 14)
        self.fc_projection2 = torch.nn.Linear(14, 14)

        self.fc_join = torch.nn.Linear(25, 14)
        self.fc_join2 = torch.nn.Linear(14, 14)

        self.fc_selection = torch.nn.Linear(16, 14)
        self.fc_selection2 = torch.nn.Linear(14, 14)

        self.rnn = torch.nn.GRU(14, hidden_size, num_layers, bidirectional=bidirectional)

        self.fc = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), 18)
        self.fc2 = torch.nn.Linear(18, 18)
        self.fc_out = torch.nn.Linear(18, 1)

    def forward(self, x):
        embedded_seq = []
        for step in x:
            step = step.view(-1)
            if len(step) == 14:
                embedded = self.fc_projection2(F.tanh(self.fc_projection(step)))
            elif len(step) == 25:
                embedded = self.fc_join2(F.tanh(self.fc_join(step)))
            elif len(step) == 16:
                embedded = self.fc_selection2(F.tanh(self.fc_selection(step)))
            else:
                raise ValueError(f'Unexpected length {len(step)}.\n{step}')

            embedded = torch.tanh(embedded)
            embedded_seq.append(embedded)

        embedded_seq = torch.stack(embedded_seq)
        seq_len, vec_len = embedded_seq.size()
        out, _ = self.rnn(embedded_seq.view(seq_len, 1, vec_len))
        out = out[-1, :, :]
        out = torch.tanh(out)

        out = self.fc(out)
        out = torch.tanh(out)

        out = self.fc2(out)
        out = torch.tanh(out)

        return self.fc_out(out)

    def training_step(self, batch, batch_idx):
        x, y = batch['input'], batch['runtime']
        y_hat = self(x)
        loss = F.smooth_l1_loss(y_hat, y)
        self.log('train_loss', loss)

        return loss

    def on_validation_epoch_start(self):
        self.validation_predictions = []
        self.validation_targets = []

    def validation_step(self, batch, batch_idx):
        x, y = batch['input'], batch['runtime']
        y_hat = self(x)
        loss = F.smooth_l1_loss(y_hat, y)
        self.log('val_loss', loss)

        self.validation_predictions.append(np.round(y_hat.item()))
        self.validation_targets.append(np.round(y.item()))

        return loss

    def on_validation_epoch_end(self, **outputs):
        if len(self.validation_predictions) > 10:
            ge.save_log_results(
                self.validation_predictions,
                self.validation_targets,
                path=self.trainer.logger.log_dir,
                prefix=f'val_results_{self.current_epoch}'
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
