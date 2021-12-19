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


class NodeEmbedding(torch.nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.drop = torch.nn.Dropout(dropout)
        self.relation_embedding_1 = torch.nn.Linear(63, 53)
        self.relation_embedding_2 = torch.nn.Linear(53, 48)
        self.relation_embedding_3 = torch.nn.Linear(48, 48)

        self.projection_embedding_1 = torch.nn.Linear(69, 59)
        self.projection_embedding_2 = torch.nn.Linear(59, 48)
        self.projection_embedding_3 = torch.nn.Linear(48, 48)

        self.selection_embedding_1 = torch.nn.Linear(112, 98)
        self.selection_embedding_2 = torch.nn.Linear(98, 80)
        self.selection_embedding_3 = torch.nn.Linear(80, 64)
        self.selection_embedding_4 = torch.nn.Linear(64, 48)

        self.join_embedding_1 = torch.nn.Linear(203, 175)
        self.join_embedding_2 = torch.nn.Linear(175, 150)
        self.join_embedding_3 = torch.nn.Linear(150, 128)
        self.join_embedding_4 = torch.nn.Linear(128, 80)
        self.join_embedding_5 = torch.nn.Linear(80, 48)

    def forward(self, x):
        x = x.view(-1)

        if len(x) == 63:
            x = torch.tanh(self.drop(self.relation_embedding_1(x)))
            x = torch.tanh(self.relation_embedding_2(x))
            x = self.relation_embedding_3(x)
        elif len(x) == 69:
            x = torch.tanh(self.drop(self.projection_embedding_1(x)))
            x = torch.tanh(self.projection_embedding_2(x))
            x = self.projection_embedding_3(x)
        elif len(x) == 112:
            x = torch.tanh(self.drop(self.selection_embedding_1(x)))
            x = torch.tanh(self.selection_embedding_2(x))
            x = torch.tanh(self.selection_embedding_3(x))
            x = self.selection_embedding_4(x)
        elif len(x) == 203:
            x = torch.tanh(self.drop(self.join_embedding_1(x)))
            x = torch.tanh(self.join_embedding_2(x))
            x = torch.tanh(self.join_embedding_3(x))
            x = torch.tanh(self.join_embedding_4(x))
            x = self.join_embedding_5(x)
        else:
            raise ValueError(f'Unexpected length {len(x)}.\n{x}')

        return x


class GereltRegressor(LightningModule):
    """
    Generic Embedding for RELational Tree
    """

    def __init__(self, hidden_size=125, num_layers=4, bidirectional=True, optimizer: str = None):
        super().__init__()
        self.optim = optimizer
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.node_embedding = NodeEmbedding()
        self.rnn = torch.nn.GRU(48, hidden_size, num_layers, bidirectional=bidirectional, dropout=0.1)

        self.fc = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), 48)
        self.fc2 = torch.nn.Linear(48, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc_out = torch.nn.Linear(16, 1)

    def forward(self, x):
        embedded_seq = []
        self.node_embedding.to(self.device)
        for node in x:
            node.to(self.device)
            embedded_seq.append(torch.tanh(self.node_embedding(node)))

        embedded_seq = torch.stack(embedded_seq)
        seq_len, vec_len = embedded_seq.size()
        out, _ = self.rnn(embedded_seq.view(seq_len, 1, vec_len))
        out = out[-1, :, :]
        out = torch.tanh(out)

        out = self.fc(out)
        out = torch.tanh(out)

        out = self.fc2(out)
        out = torch.tanh(out)

        out = self.fc3(out)
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
        if self.optim and self.optim == 'adadelta':
            return torch.optim.Adadelta(self.parameters())
        else:
            return torch.optim.Adam(self.parameters(), lr=1e-4)
