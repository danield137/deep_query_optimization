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
from torch.nn import functional as F

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


def save_confusion_figure(predictions, targets, fpath, fname):
    plt.clf()

    predictions = np.array(predictions)
    actual = np.array(targets)

    mae = np.mean(np.abs(predictions - targets))
    
    labels = list(set(list(actual)) | set(list(predictions)))
    labels = [str(l) for l in sorted(labels)]
    
    conf = sklearn.metrics.confusion_matrix(predictions, actual)
    acc = sklearn.metrics.accuracy_score(predictions, actual)
    balanced_acc = sklearn.metrics.balanced_accuracy_score(predictions, actual, adjusted=True)
    f1_m = sklearn.metrics.f1_score(predictions, actual, average='macro')
    f1_w = sklearn.metrics.f1_score(predictions, actual, average='weighted')
    
    sns.heatmap(conf, fmt="g", annot=True, xticklabels=labels, yticklabels=labels)
    plt.title(f'accuracy: {acc} \n balanced_accuracy: {balanced_acc} \n mae: {np.mean(np.abs(actual - predictions))} \n f1 macro: {f1_m} \n f1 weighted: {f1_w}')

    save_path = os.path.join(fpath, fname + f'_acc_{acc}_mae_{mae}_f1_{f1_m}.png')
    plt.savefig(save_path, bbox_inches='tight')

JOIN_VEC_LEN = 14
PROJECTION_VEC_LEN = 25
SELECTION_VEC_LEN = 16

class GREQRegressor(LightningModule):
    """
    Generic Relational Query Classifier
    """

    def __init__(self, hidden_size=90, num_layers=2, max_seq=100, bidirectional=True):
        super().__init__()
        self.max_seq = max_seq
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.fc_projection = torch.nn.Linear(14, 14)
        self.fc_join = torch.nn.Linear(25, 14)
        self.fc_selection = torch.nn.Linear(16, 14)
        self.rnn = torch.nn.GRU(14, hidden_size, num_layers, bidirectional=bidirectional)

        self.fc = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 18)
        self.fc_out = torch.nn.Linear(18, 1)

    def forward(self, x):
        embedded_seq = []
        for step in x:
            step = step.view(-1)
            if len(step) == 14:
                embedded = self.fc_projection(step)
            elif len(step) == 25:
                embedded = self.fc_join(step)
            elif len(step) == 16:
                embedded = self.fc_selection(step)
            else:
                raise ValueError(f'Unexpected length {len(step)}.\n{step}')

            embedded = F.tanh(embedded)
            embedded_seq.append(embedded)

        embedded_seq = torch.stack(embedded_seq)
        seq_len, vec_len = embedded_seq.size()
        out, _ = self.rnn(embedded_seq.view(seq_len, 1, vec_len))
        out = out[-1, :, :]
        out = F.tanh(out)

        out = self.fc(out)
        out = F.tanh(out)

        out = self.fc2(out)
        out = F.tanh(out)

        out = self.fc3(out)
        out = F.tanh(out)

        return self.fc_out(out)

    def training_step(self, batch, batch_idx):
        x, y = batch['input'], batch['runtime']
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        result = pl.TrainResult(loss)
        result.log('train_loss', loss)

        return result

    def on_validation_epoch_start(self):
        self.validation_predictions = []
        self.validation_targets = []

    def validation_step(self, batch, batch_idx):
        x, y = batch['input'], batch['runtime']
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult(loss, checkpoint_on=loss)
        result.log('val_loss', loss)

        self.validation_predictions.append(np.round(y_hat.item()))
        self.validation_targets.append(np.round(y.item()))

        return result

    def on_validation_epoch_end(self, **outputs):
        if len(self.validation_predictions) > 10:
            save_confusion_figure(self.validation_predictions, self.validation_targets, self.trainer.logger.log_dir,
                                  f'val_conf_matrix_{self.current_epoch}.png')

    def on_test_epoch_start(self):
        self.test_targets = []
        self.test_predictions = []

    def test_step(self, batch, batch_idx):
        x, y = batch['input'], batch['runtime']
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult()
        result.log('test_loss', loss)

        self.test_predictions.append(np.round(y_hat.item()))
        self.test_targets.append(np.round(y.item()))

        return result

    def on_test_epoch_end(self):
        if len(self.test_predictions) > 10:
            save_confusion_figure(self.test_predictions, self.test_targets, self.trainer.logger.log_dir, f'test_conf_matrix_{self.current_epoch}')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)
