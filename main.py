"""Standard training script"""

import pytorch_lightning as pl
from model import VAE
from data import datamodule


pl.seed_everything(42)
vae = VAE()
trainer = pl.Trainer(gpus=1, max_epochs=10)
trainer.fit(vae, datamodule)