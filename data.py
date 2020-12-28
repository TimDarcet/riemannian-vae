"""Hold data loader / dataset / datamodule definitions"""

from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule


datamodule = MNISTDataModule('./data')
