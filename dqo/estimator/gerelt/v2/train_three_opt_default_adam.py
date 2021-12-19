import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from dqo.estimator.gerelt.v2.data_module import QueriesDataModule
from dqo.estimator.gerelt.v2.model import GereltRegressor
from dqo.log_utils import enable_dqo_logs

if __name__ == '__main__':
    enable_dqo_logs()
    seed_everything(42)
    split_kind = 'default'
    tb_logger = pl_loggers.TensorBoardLogger('logs/', name="three_opt_default_adam")
    conf = dict(
        auto_select_gpus=True,
        checkpoint_callback=ModelCheckpoint(save_top_k=-1),
        gradient_clip_val=0.5,
        auto_lr_find=False,
        min_epochs=100,
        max_epochs=250,
        logger=tb_logger,
        progress_bar_refresh_rate=50
    )

    if torch.cuda.is_available():
        conf['gpus'] = 1
    else:
        conf['max_epochs'] = 3
        conf['max_steps'] = 500

    trainer = pl.Trainer(**conf)

    queries_dm = QueriesDataModule(['imdb:optimized', 'tpch:optimized', 'tpcds:optimized'], split_kind=split_kind)
    queries_dm.prepare_data()
    queries_dm.setup('fit')

    model = GereltRegressor()
    trainer.fit(model, datamodule=queries_dm)
