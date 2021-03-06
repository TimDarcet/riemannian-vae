"""Standard training script"""

import pytorch_lightning as pl
from model import VAE
from data import datamodule


pl.seed_everything(42)
vae = VAE(latent_dim=256, input_height=32, input_channels=1)
trainer = pl.Trainer(gpus=1, max_epochs=1, progress_bar_refresh_rate=20)
trainer.fit(vae, datamodule)
