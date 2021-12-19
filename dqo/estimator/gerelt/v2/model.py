import matplotlib as mpl
import numpy as np
import torch

mpl.use('Agg')
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F
from dqo.estimator import evaluate as ge

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


class GereltRegressor(LightningModule):
    """
    Generic Embedding for RELational Tree
    """

    def __init__(self, hidden_size=125, num_layers=2, bidirectional=True, optimizer: str = None):
        super().__init__()
        self.optim = optimizer
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.relation_embedding_1 = torch.nn.Linear(49, 32)
        self.relation_embedding_2 = torch.nn.Linear(32, 32)
        self.relation_embedding_3 = torch.nn.Linear(32, 32)

        self.projection_embedding_1 = torch.nn.Linear(53, 32)
        self.projection_embedding_2 = torch.nn.Linear(32, 32)
        self.projection_embedding_3 = torch.nn.Linear(32, 32)

        self.selection_embedding_1 = torch.nn.Linear(98, 64)
        self.selection_embedding_2 = torch.nn.Linear(64, 32)
        self.selection_embedding_3 = torch.nn.Linear(32, 32)

        self.join_embedding_1 = torch.nn.Linear(175, 128)
        self.join_embedding_2 = torch.nn.Linear(128, 64)
        self.join_embedding_3 = torch.nn.Linear(64, 32)

        self.rnn = torch.nn.GRU(32, hidden_size, num_layers, bidirectional=bidirectional)

        self.fc = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc_out = torch.nn.Linear(16, 1)
        self.drop = torch.nn.Dropout(.2)

    def forward(self, x):
        embedded_seq = []
        for step in x:
            step = step.view(-1).to(self.device)

            if len(step) == 49:
                embedded = self.drop(torch.tanh(self.relation_embedding_1(step)))
                embedded = torch.tanh(self.relation_embedding_2(embedded))
                embedded = torch.tanh(self.relation_embedding_3(embedded))
            elif len(step) == 53:
                embedded = self.drop(torch.tanh(self.projection_embedding_1(step)))
                embedded = torch.tanh(self.projection_embedding_2(embedded))
                embedded = torch.tanh(self.projection_embedding_3(embedded))
            elif len(step) == 98:
                embedded = self.drop(torch.tanh(self.selection_embedding_1(step)))
                embedded = torch.tanh(self.selection_embedding_2(embedded))
                embedded = torch.tanh(self.selection_embedding_3(embedded))
            elif len(step) == 175:
                embedded = self.drop(torch.tanh(self.join_embedding_1(step)))
                embedded = torch.tanh(self.join_embedding_2(embedded))
                embedded = torch.tanh(self.join_embedding_3(embedded))
            else:
                raise ValueError(f'Unexpected length {len(step)}.\n{step}')

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

        self.validation_predictions.append(y_hat.item())
        self.validation_targets.append(y.item())

        return loss

    def on_validation_epoch_end(self, **outputs):
        if len(self.validation_predictions) > 10:
            ge.save_log_results(
                self.validation_predictions,
                self.validation_targets,
                path=self.trainer.logger.log_dir,
                prefix=f'val_results_{self.current_epoch}'
            )

    def test_step(self, batch, batch_idx):
        x, y = batch['input'], batch['runtime']
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)

        return loss

    def configure_optimizers(self):
        if self.optim and self.optim == 'adadelta':
            return torch.optim.Adadelta(self.parameters())
        else:
            return torch.optim.Adam(self.parameters(), lr=1e-4)
