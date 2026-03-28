from config import TIMITConfig
from argparse import ArgumentParser
from multiprocessing import Pool
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import torch
import torch.utils.data as data
import random
import numpy as np

# SEED
def seed_torch(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed, workers=True)

seed_torch()

from TIMIT.dataset import TIMITDataset
from TIMIT.lightning_model_uncertainty_loss import LightningModel

import torch.nn.utils.rnn as rnn_utils

def collate_fn(batch):
    (seq, height, age, gender) = zip(*batch)
    seql = [x.reshape(-1,) for x in seq]
    seq_length = [x.shape[0] for x in seql]
    data_ = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    return data_, height, age, gender, seq_length

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default=TIMITConfig.data_path)
    parser.add_argument('--speaker_csv_path', type=str, default=TIMITConfig.speaker_csv_path)
    parser.add_argument('--batch_size', type=int, default=TIMITConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=TIMITConfig.epochs)
    parser.add_argument('--num_layers', type=int, default=TIMITConfig.num_layers)
    parser.add_argument('--feature_dim', type=int, default=TIMITConfig.feature_dim)
    parser.add_argument('--lr', type=float, default=TIMITConfig.lr)
    parser.add_argument('--gpu', type=int, default=TIMITConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=TIMITConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=TIMITConfig.model_checkpoint)
    parser.add_argument('--model_type', type=str, default=TIMITConfig.model_type)
    parser.add_argument('--upstream_model', type=str, default=TIMITConfig.upstream_model)
    parser.add_argument('--narrow_band', type=str, default=TIMITConfig.narrow_band)

    hparams = parser.parse_args()

    # Check device
    if not torch.cuda.is_available():
        hparams.gpu = 0
    else:
        print(f'Training Model on TIMIT Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # Training, Validation and Testing Dataset
    # Training Dataset
    train_set = TIMITDataset(
        wav_folder=os.path.join(hparams.data_path, 'TRAIN'),
        hparams=hparams
    )
    # Training DataLoader
    trainloader = data.DataLoader(
        train_set,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=hparams.n_workers,
        collate_fn=collate_fn,
    )
    # Validation Dataset
    valid_set = TIMITDataset(
        wav_folder=os.path.join(hparams.data_path, 'VAL'),
        hparams=hparams,
        is_train=False
    )
    # Validation Dataloader
    valloader = data.DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=hparams.n_workers,
        collate_fn=collate_fn,
    )
    # Testing Dataset
    test_set = TIMITDataset(
        wav_folder=os.path.join(hparams.data_path, 'TEST'),
        hparams=hparams,
        is_train=False
    )
    # Testing Dataloader
    testloader = data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=hparams.n_workers,
        collate_fn=collate_fn,
    )

    print('Dataset Split (Train, Validation, Test)=', len(train_set), len(valid_set), len(test_set))

    logger = WandbLogger(
        name=TIMITConfig.run_name,
        project='SpeakerProfiling'
    )

    model = LightningModel(vars(hparams))

    model_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        monitor='val/loss',
        mode='min',
        verbose=1)

    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    devices = hparams.gpu if (torch.cuda.is_available() and hparams.gpu and hparams.gpu > 0) else 1
    strategy = "ddp" if (torch.cuda.is_available() and devices and devices > 1) else "auto"

    trainer = pl.Trainer(
        fast_dev_run=bool(hparams.dev),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=hparams.epochs,
        callbacks=[
            EarlyStopping(
                monitor='val/loss',
                min_delta=0.00,
                patience=10,
                verbose=True,
                mode='min'
            ),
            model_checkpoint_callback
        ],
        logger=logger,
        auto_lr_find=True
    )

    # Fit model
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader, ckpt_path=hparams.model_checkpoint)

    print('\n\nCompleted Training...\nTesting the model with checkpoint -', model_checkpoint_callback.best_model_path)