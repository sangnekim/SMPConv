##  SMPConv: Self-moving Point Representations for Continuous Convolution

This directory includes **image classification with ImageNet-1k**

## Results

| model | #params | FLOPs | acc@1 | pretrained |
|:---:|:---:|:---:|:---:|:---:|
| SMPConv-T | 27M | 5.7G | 82.5 | [Google Drive](https://drive.google.com/file/d/1xT9Y4cAj4I6rmNJ28-Mh4_APcsEqeJ1V/view) |
| SMPConv-B | 80M | 16.6G | 83.8 | [Google Drive](https://drive.google.com/file/d/16gd2KFnK1fgdEpOulnCYo2KaPl547zT9/view) |


## Training

We trained our models with 4 A100 80GB GPUs  
Batch size: batch_size x update_freq x num_GPUs

### ImageNet-1K SMPConv-T
 * batch size: 256 x 2 x 4 = 2048
 * drop path rate: 0.1
```
python -m torch.distributed.launch --nproc_per_node=4 main.py  \
--epochs 300 --model SMPConv_T --batch_size 256 --warmup_epochs 10 \
--lr 4e-3 --update_freq 2 --model_ema true --model_ema_eval true \
--data_dir /path/to/imagenet-1k --num_workers 16
```


### ImageNet-1K SMPConv-B
 * batch size: 128 x 4 x 4 = 2048  
 * drop path rate: 0.5
```
python -m torch.distributed.launch --nproc_per_node=4 main.py  \
--epochs 300 --model SMPConv_B --batch_size 128 --warmup_epochs 10 \
--lr 4e-3 --update_freq 4 --model_ema true --model_ema_eval true \
--data_dir /path/to/imagenet-1k --num_workers 16
```

## Evaluation
```
python main.py --model SMPConv_T --eval true \
--model_ema true --model_ema_eval true --resume /path/to/checkpoint \
--batch_size 256 --num_workers 16 --data_dir /path/to/imagenet-1k
```
