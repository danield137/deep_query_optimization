import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from dqo.estimator.gerelt.v15.data_module import QueriesDataModule
from dqo.estimator.gerelt.v15 import model, encoder
from dqo.log_utils import enable_dqo_logs

if __name__ == '__main__':
    enable_dqo_logs()
    seed_everything(42)

    tb_logger = pl_loggers.TensorBoardLogger('logs/', name="tpch")
    conf = dict(
        auto_select_gpus=False,
        checkpoint_callback=ModelCheckpoint(save_top_k=-1,period=1),
        auto_lr_find=False,
        min_epochs=200,
        max_epochs=250,
        logger=tb_logger,
        gradient_clip_val=1.5,
        progress_bar_refresh_rate=50
    )

    if torch.cuda.is_available():
        conf['gpus'] = 1
    else:
        conf['max_epochs'] = 3
        conf['max_steps'] = 800

    trainer = pl.Trainer(**conf)

    queries_dm = QueriesDataModule(['tpch:optimized:train'], encoder, split_kind='default')
    queries_dm.prepare_data()
    queries_dm.setup('fit')

    trainer.fit(model.GereltRegressor(), datamodule=queries_dm)
