import matplotlib as mpl
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
        self.relation_embedding_1 = torch.nn.Linear(63, 48)
        self.relation_embedding_2 = torch.nn.Linear(48, 48)

        self.projection_embedding_1 = torch.nn.Linear(69, 48)
        self.projection_embedding_2 = torch.nn.Linear(48, 48)

        self.selection_embedding_1 = torch.nn.Linear(49, 48)
        self.selection_embedding_2 = torch.nn.Linear(48, 48)

        self.join_embedding_1 = torch.nn.Linear(77, 64)
        self.join_embedding_2 = torch.nn.Linear(64, 48)

    def forward(self, x):
        x = x.view(-1)

        if len(x) == 63:
            x = torch.tanh(self.drop(self.relation_embedding_1(x)))
            x = self.relation_embedding_2(x)
        elif len(x) == 69:
            x = torch.tanh(self.drop(self.projection_embedding_1(x)))
            x = self.projection_embedding_2(x)
        elif len(x) == 49:
            x = torch.tanh(self.drop(self.selection_embedding_1(x)))
            x = self.selection_embedding_2(x)
        elif len(x) == 77:
            x = torch.tanh(self.drop(self.join_embedding_1(x)))
            x = self.join_embedding_2(x)
        else:
            raise ValueError(f'Unexpected length {len(x)}.\n{x}')

        return x


class CNNRegressor(LightningModule):
    def __init__(self, hidden_size=125, num_layers=2, bidirectional=True, optimizer: str = None):
        super().__init__()
        self.node_embed = NodeEmbedding()
        # max depth is 125
        # size 125x48
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 48, 3, stride=3),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(3),
            torch.nn.Conv2d(48, 512, 3, stride=3),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d((4, 1)),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 16),

        )
        self.fc_out = torch.nn.Linear(16, 1)

    def forward(self, x):
        embedded_seq = []
        self.node_embed.to(self.device)
        for node in x:
            node.to(self.device)

            embedded_seq.append(self.node_embed(node))

        embedded_seq = torch.stack(embedded_seq)
        seq_len, vec_len = embedded_seq.size()
        embedded_seq = torch.nn.ZeroPad2d((0, 0, 0, 125 - seq_len))(embedded_seq)
        embedded_seq = embedded_seq.unsqueeze(0).unsqueeze(0)
        out = torch.tanh(self.cnn(embedded_seq))

        out = out.view(-1)

        out = self.fc(out)
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
