import time

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from dqo.estimator.greq.regressor_v2.data_module import QueriesDataModule
from dqo.estimator.greq.regressor_v2.model import GREQRegressor

if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('logs/', name="imdb_aug")
    conf = dict(
        auto_select_gpus=True,
        checkpoint_callback=ModelCheckpoint(save_top_k=-1),
        early_stop_callback=EarlyStopping(
            patience=10
        ),
        auto_lr_find=False,
        min_epochs=200,
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

    queries_dm = QueriesDataModule(['imdb:resp_time_aug'])
    queries_dm.prepare_data()
    queries_dm.setup('fit')

    model = GREQRegressor(max_seq=100)
    trainer.fit(model, datamodule=queries_dm)

    # trainer.save_checkpoint("manual.ckpt", weights_only=True)

    # torch.save(model, f'regressor_pickle_{int(time.time())}.pth')
    # better save
    # m = torch.jit.script(model)
    # m.save('regressor.pth')
