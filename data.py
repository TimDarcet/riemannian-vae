"""Hold data loader / dataset / datamodule definitions"""

from pl_bolts.datamodules import CIFAR10DataModule
from resized_mnist_datamodule import ResizedMNISTDataModule
from torchvision import transforms


# datamodule = CIFAR10DataModule('./data')
datamodule = ResizedMNISTDataModule('./data')
