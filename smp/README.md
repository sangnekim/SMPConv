## SMPConv: Self-moving Point Representations for Continuous Convolution

This directory includes **sequential data classifications** and **image classification with CIFAR10**

## Results

| Data | #params | kernel size | acc@1 | model|
|:---:|:---:|:---:|:---:|:---:|
| sMNIST | 371k | 784 | 99.75 | [Google Drive](https://drive.google.com/file/d/164qRzZs1kvuoOr34FLukGIQAI7PAYWUf/view) |
| pMNIST | 371k | 784 | 99.10 | [Google Drive](https://drive.google.com/file/d/1no63cSqPgSJIkW8-iSvrAA1pFlurOKsh/view) |
| sCIFAR10 | 373k | 1024 | 84.86 | [Google Drive](https://drive.google.com/file/d/1ilu1Oo74B4bC8cjkoIavPqJmEK34PhYg/view) |
| Character Trajectories | 374k | 182 | 99.53 | [Google Drive](https://drive.google.com/file/d/1Jxy00LiTdjTXa-kYH7cJYX2s1A2yJSPK/view) |
| Speech Commands | 392k | 161 | 97.71 | [Google Drive](https://drive.google.com/file/d/1AKmeyBDkv0v5qk8GF4vkecY6PgR2JpGP/view) |
| Speech Commands (raw) | 371k | 16000 | 94.95 | [Google Drive](https://drive.google.com/file/d/15OWR-ZZIVu12pTtdCU1Dyv87yxwIhRlu/view) |
| CIFAR10 | 490k | 33 x 33 | 93.00 | [Google Drive](https://drive.google.com/file/d/1aKwORBf-lplfdwB8WEcBRRAL7IA4ejth/view) |


## Train
**Requirements are [here](https://github.com/sangnekim/SMPConv#requirements)**
#### sMNIST
```
python run_experiment.py conv.horizon=same train.batch_size=64 net.no_blocks=6 net.no_hidden=30 conv.type=CKConv dataset=sMNIST device=cuda net.dropout_in=0.1 train.epochs=200 kernel.n_points=30 kernel.radius=0.002 kernel.coord_std=0.1 conv.small_kernel_size=5 net.type=TCN net.norm=BatchNorm train.radius_lr_factor=0.1 train.augment=standard train.optimizer=Adam train.scheduler=cosine train.weight_decay=1e-5 summary="[64, 1, 784]" seed=0 conv.use_fft=True dataset_params.permuted=False net.dropout=0 train.lr=0.0001
```

#### pMNIST
```
python run_experiment.py conv.horizon=same train.batch_size=64 net.no_blocks=6 net.no_hidden=30 conv.type=CKConv dataset=sMNIST device=cuda net.dropout_in=0.1 train.epochs=200 kernel.n_points=30 kernel.radius=0.002 kernel.coord_std=0.1 conv.small_kernel_size=5 net.type=TCN net.norm=BatchNorm train.radius_lr_factor=0.1 train.augment=standard train.optimizer=Adam train.scheduler=cosine train.weight_decay=1e-5 summary="[64, 1, 784]" seed=0 conv.use_fft=True dataset_params.permuted=True net.dropout=0 train.lr=0.0001
```

#### sCIFAR10
```
python run_experiment.py conv.horizon=same conv.bias=True train.batch_size=64 net.no_blocks=6 net.no_hidden=30 conv.type=CKConv dataset=sCIFAR10 device=cuda net.dropout_in=0.0 train.epochs=200 kernel.n_points=30 kernel.radius=0.002 kernel.coord_std=0.1 conv.small_kernel_size=5 net.type=TCN net.norm=BatchNorm train.radius_lr_factor=0.1 train.augment=resnet train.optimizer=Adam train.scheduler=cosine train.weight_decay=1e-5 summary="[64, 3, 1024]" seed=0 conv.use_fft=True net.dropout=0 train.lr=0.0002
```

#### Character Trajectories
```
python run_experiment.py conv.horizon=same conv.bias=True train.batch_size=64 net.no_blocks=6 net.no_hidden=30 conv.type=CKConv dataset=CharTrajectories device=cuda net.dropout_in=0.0 train.epochs=300 kernel.n_points=30 kernel.radius=0.012 kernel.coord_std=0.1 conv.small_kernel_size=5 net.type=TCN net.norm=BatchNorm train.radius_lr_factor=0.1 train.optimizer=Adam train.scheduler=cosine train.weight_decay=1e-5 summary="[64, 3, 182]" seed=0 dataset_params.mfcc=True conv.use_fft=False net.dropout=0 train.lr=0.0001
```

#### Speech Commands
```
python run_experiment.py conv.horizon=same conv.bias=True train.batch_size=32 net.no_blocks=6 net.no_hidden=30 conv.type=CKConv dataset=SpeechCommands device=cuda net.dropout=0.2 net.dropout_in=0.0 train.epochs=300 kernel.n_points=30 kernel.radius=0.012 kernel.coord_std=0.1 conv.small_kernel_size=5 net.type=TCN net.norm=BatchNorm train.radius_lr_factor=0.1 train.optimizer=Adam train.scheduler=cosine train.weight_decay=1e-5 summary="[32, 20, 161]" seed=0 dataset_params.mfcc=True conv.use_fft=False train.lr=0.002
```

#### Speech Commands Raw
```
python run_experiment.py conv.horizon=same conv.bias=False train.batch_size=32 net.no_blocks=6 net.no_hidden=30 conv.type=CKConv dataset=SpeechCommands device=cuda net.dropout_in=0.0 train.epochs=160 kernel.n_points=30 kernel.radius=0.0002 kernel.coord_std=0.1 conv.small_kernel_size=5 net.type=TCN net.norm=BatchNorm train.radius_lr_factor=0.1 train.optimizer=Adam train.scheduler=cosine train.scheduler_params.warmup_epochs=10 train.weight_decay=1e-5 summary="[32, 1, 16000]" dataset_params.mfcc=False conv.use_fft=True seed=0 net.dropout=0.1 train.lr=0.001
```

#### CIFAR10
```
python run_experiment.py conv.horizon=33 train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" net.no_blocks=7 net.no_hidden=24 conv.type=CKConv dataset=CIFAR10 device=cuda net.dropout_in=0 train.epochs=210 kernel.radius=0.12 kernel.coord_std=0.05 conv.small_kernel_size=3 train.radius_lr_factor=0.1 net.type=ResNet net.norm=BatchNorm train.augment=resnet train.optimizer=Adam train.scheduler=cosine train.scheduler_params.warmup_epochs=10 train.weight_decay=1e-5 summary=[64,3,32,32] seed=0 kernel.n_points=16 net.dropout=0.1 train.lr=0.005
```
## Evaluation
To evaluate pretrained model, add this command on above train command.
```
pretrained=True pretrained_params.filepath=/path/to/pretrained_params test.before_train=True debug=True
```
