import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy
import argparse
import time
import os


class CIFAR10Classifier(LightningModule):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        # init the pretrained LightningModule
        self.net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.net.fc = nn.Linear(512, 10)
        # for param in self.net.parameters():
        #     param.requires_grad = False
        # for param in self.net.fc.parameters():
        #     param.requires_grad = True
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.val_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, y)

        self.log("train_loss",  loss, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True, on_step=False, on_epoch=True)
    
    def prepare_data(self) -> None:
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    
    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.train_transform)
            train_set_len = int(len(trainset_full) * 0.7)
            self.train_set, self.val_set = torch.utils.data.random_split(trainset_full, [train_set_len, len(trainset_full) - train_set_len])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.val_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.args.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.args.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.args.batch_size)

def parse_args():
    parser = argparse.ArgumentParser(description='Training baseline model')

    parser.add_argument('--exp_name', type=str, default="base_line", help="name of the experiment that is used as dir name")
    parser.add_argument("--output_dir", type=str, default=".\experiment_results")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    time_stamp = time.strftime("-%Y-%m-%d-%H-%M", time.localtime())
    args.exp_name = args.exp_name + time_stamp
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # set model
    model = CIFAR10Classifier(args)
    
    # training
    trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=args.epochs,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=TensorBoardLogger(args.output_dir, name=args.exp_name)
    )

    trainer.fit(model)