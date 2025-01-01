<h1 align="center">
  <b>ResNet-50 on ImageNet-1K</b>
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-31012/">
    <img src="https://img.shields.io/badge/Python-3.10.12-blue" alt="Python 3.10.12">
  </a>
  <a href="Static Badge">
    <img src="https://img.shields.io/badge/PyTorch-2.5.1-red" alt="PyTorch-2.5.1"/>
  </a>
  <a href="https://huggingface.co/spaces/tranquilkd/ResNet-50-ImageNet-1K">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces"/>
  </a>
</p>

<p align="center">
Train ResNet-50 on ImageNet-1K dataset with test accuracy for top-1 should at least be 70%.
</p>


# [Hugging Face Demo](https://huggingface.co/spaces/tranquilkd/ResNet-50-ImageNet-1K)

![Hugging Face Demo](<assets/huggingface space demo.png>)


# DATASET

1. EBS volume creation on AWS to store the dataset
2. Mount this volume to any AMI for downloading the dataset
    *   Create an instance using spot requests (no need to have a GPU)
    *   Connect to your instance using SSH.
    *   Use the lsblk command to view your available disk devices and their mount points, looks something like this:
        ```bash
        [ec2-user@ip-172-31-86-46 ~]$ lsblk
        NAME    MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
        xvda    202:0    0    8G  0 disk
        └─xvda1 202:1    0    8G  0 part /
        xvdb    202:16   0    8G  0 disk
        xvdf    202:80   0  400G  0 disk
        ```
    *   Create a file system on the volume, example -> `sudo mkfs -t ext4 /dev/xvdf`
    *   Create a mount point directoty for the volume -> `sudo mkdir /data`
    *   To mount this EBS volume at the location you just created -> `sudo mount /dev/xvdf /data`
    *   To check you can perform `ls /data`
3. To download dataset: 
    *   `pip install kaggle` to anywhere as this is temp env just for downloading the data so, do not need venv.
    *   Then copy your `kaggle.json` to `~/.kaggle/kaggle.json` OR 
        ```bash
        export KAGGLE_USERNAME=datadinosaur
        export KAGGLE_KEY=xxxxxxxxxxxxxx
        ```
    *   Then `kaggle competitions download -c imagenet-object-localization-challenge` this will download dataset from here https://www.kaggle.com/c/imagenet-object-localization-challenge/data
    *   Once downloaded we need to extract the data using `unzip imagenet-object-localization-challenge.zip -d imagenet`
    ```text
    imagenet
    ├── ILSVRC
    │   ├── Annotations
    │   │   └── CLS-LOC
    │   │       ├── train
    │   │       └── val
    │   ├── Data    
    │   │   ├── CLS-LOC
    │   │   │   ├── test                   # test images
    │   │   │   └── train                  # train images
    |   |   |   └── val                    # validation images
    │   └── Imagesets
    │       └── CLS-LOC
    │           ├── test.txt
    │           ├── train_cls.txt
    |           ├── train_loc.txt
    |           └── val.txt
    ├── LOC_sample_submission.csv
    ├── LOC_synset_mapping.txt
    ├── LOC_train_solution.csv
    └── LOC_val_solution.csv
    ```
4. Your data is ready to be used, now we can unmount the volume and terminate the instance then create the desired instance with sufficient GPUs to train the model.
    *   Unmount the volume `sudo umount -d /dev/xvdf`
    *   Then go to volume in the navigation page on the console and choose Actions, Detach volume.
    *   When prompted for confirmation, choose Detach.
5. Now to make this volume available for our new instance with GPU, perform this steps in the new instance
    *   Use the lsblk command to view your available disk devices and their mount points.
    *   Determine whether there is a file system on the volume using `sudo file -s /dev/xvdf`
    *   If you discovered that there is a file system and already has data on it then **DO NOT USE** `sudo mkfs -t xfs /dev/xvdf`
    *   Use the mkdir command `sudo mkdir /data` to create a mount point directory for the volume.
    *   Mount the volume or partition at the mount point directory you created using `sudo mount /dev/xvdf /data`


# EXECUTION ENV

| **Env Attribute**     | **Details**        |
|-----------------------|--------------------|
| **AWS Instance**      | g6.12xlarge        |
| **vCPUs**             | 48                 |
| **Memory**            | 192 GB             |
| **AMI name**          | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5.1 (Ubuntu 22.04) 20241208** | 
| **GPU Model**         | NVIDIA L4 24GB     |
| **GPUs**              | 4                  |
| **NN Architecture**   | ResNet-50          |
| **Dataset**           | ImageNet-1K        |
| **Framework**         | PyTorch 2.5.1      |
| **Python**            | 3.10.12            |

1. Create a virtual env using python `python3.10 -m venv .venv`
2. Activate the venv using `source .venv/bin/activate`
2. Use **requirement.txt** to download and install python packages required for the task using:
    ```bash
    python -m pip install --no-cache-dir -r /workspace/requirement.txt --extra-index-url https://download.pytorch.org/whl/cu124
    ```
3. All the required python packages will be installed!


# RUN

```bash
torchrun --standalone --max_restarts 1 --nnodes 1 --nproc_per_node=4 main.py --amp --train \
--data-dir data/imagenet/ILSVRC/Data/CLS-LOC \
--exp exp-1 --data-workers 24 --model resnet50 --batch-sz 512 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--criterion ce-with-label-smoothing --optim sgd --lr-scheduler cosineannealinglr \
--auto-augment ta_wide --epochs 100 --random-erase 0.1 --decay 0.00002 \
--norm-weight-decay 0.0 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --val-resize-size 232 --ra-sampler --ra-reps 4
```


# [TRAINING LOGS](runs/exp-1/logs/info.log)

```
W1231 03:20:54.256000 29101 torch/distributed/run.py:793]
W1231 03:20:54.256000 29101 torch/distributed/run.py:793] *****************************************
W1231 03:20:54.256000 29101 torch/distributed/run.py:793] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, pl
ease further tune the variable for optimal performance in your application as needed.
W1231 03:20:54.256000 29101 torch/distributed/run.py:793] *****************************************
[2024-12-31 03:20:56,435 - INFO - pid-29175 - main - line-116] Namespace(data_dir='data/imagenet/ILSVRC/Data/CLS-LOC', mixup_alpha=0.2, cutmix_alpha=1.0, auto_augment='ta_wide', ra_
magnitude=9, augmix_severity=3, random_erase=0.1, val_resize_size=232, val_crop_size=224, train_crop_size=176, ra_sampler=True, ra_reps=4, batch_sz=512, data_workers=24, amp=True, t
rain=True, model='resnet50', resume=False, criterion='ce-with-label-smoothing', optim='sgd', lr_scheduler='cosineannealinglr', lr=0.5, lr_warmup_epochs=5, lr_warmup_method='linear',
 lr_warmup_decay=0.01, lr_min=0.0, decay=2e-05, norm_weight_decay=0.0, momentum=0.9, warmup_epochs=10, patience_epochs=10, epochs=100, weights=None, exp='exp-1', debug=False)
[2024-12-31 03:20:56,458 - INFO - pid-29177 - main - line-116] Namespace(data_dir='data/imagenet/ILSVRC/Data/CLS-LOC', mixup_alpha=0.2, cutmix_alpha=1.0, auto_augment='ta_wide', ra_
magnitude=9, augmix_severity=3, random_erase=0.1, val_resize_size=232, val_crop_size=224, train_crop_size=176, ra_sampler=True, ra_reps=4, batch_sz=512, data_workers=24, amp=True, t
rain=True, model='resnet50', resume=False, criterion='ce-with-label-smoothing', optim='sgd', lr_scheduler='cosineannealinglr', lr=0.5, lr_warmup_epochs=5, lr_warmup_method='linear',
 lr_warmup_decay=0.01, lr_min=0.0, decay=2e-05, norm_weight_decay=0.0, momentum=0.9, warmup_epochs=10, patience_epochs=10, epochs=100, weights=None, exp='exp-1', debug=False)
[2024-12-31 03:20:56,466 - INFO - pid-29178 - main - line-116] Namespace(data_dir='data/imagenet/ILSVRC/Data/CLS-LOC', mixup_alpha=0.2, cutmix_alpha=1.0, auto_augment='ta_wide', ra_
magnitude=9, augmix_severity=3, random_erase=0.1, val_resize_size=232, val_crop_size=224, train_crop_size=176, ra_sampler=True, ra_reps=4, batch_sz=512, data_workers=24, amp=True, t
rain=True, model='resnet50', resume=False, criterion='ce-with-label-smoothing', optim='sgd', lr_scheduler='cosineannealinglr', lr=0.5, lr_warmup_epochs=5, lr_warmup_method='linear',
 lr_warmup_decay=0.01, lr_min=0.0, decay=2e-05, norm_weight_decay=0.0, momentum=0.9, warmup_epochs=10, patience_epochs=10, epochs=100, weights=None, exp='exp-1', debug=False)
[2024-12-31 03:20:56,492 - INFO - pid-29176 - main - line-116] Namespace(data_dir='data/imagenet/ILSVRC/Data/CLS-LOC', mixup_alpha=0.2, cutmix_alpha=1.0, auto_augment='ta_wide', ra_
magnitude=9, augmix_severity=3, random_erase=0.1, val_resize_size=232, val_crop_size=224, train_crop_size=176, ra_sampler=True, ra_reps=4, batch_sz=512, data_workers=24, amp=True, t
rain=True, model='resnet50', resume=False, criterion='ce-with-label-smoothing', optim='sgd', lr_scheduler='cosineannealinglr', lr=0.5, lr_warmup_epochs=5, lr_warmup_method='linear',
 lr_warmup_decay=0.01, lr_min=0.0, decay=2e-05, norm_weight_decay=0.0, momentum=0.9, warmup_epochs=10, patience_epochs=10, epochs=100, weights=None, exp='exp-1', debug=False)
[2024-12-31 03:20:57,488 - INFO - pid-29175 - main - line-123] Preprocess data and store it..
[2024-12-31 03:20:57,488 - INFO - pid-29175 - dataset - line-234] load training data
[2024-12-31 03:20:57,840 - INFO - pid-29178 - main - line-123] Preprocess data and store it..
[2024-12-31 03:20:57,840 - INFO - pid-29178 - dataset - line-234] load training data
[2024-12-31 03:20:57,841 - INFO - pid-29177 - main - line-123] Preprocess data and store it..
[2024-12-31 03:20:57,841 - INFO - pid-29177 - dataset - line-234] load training data
[2024-12-31 03:20:57,844 - INFO - pid-29176 - main - line-123] Preprocess data and store it..
[2024-12-31 03:20:57,844 - INFO - pid-29176 - dataset - line-234] load training data
[2024-12-31 03:20:59,971 - INFO - pid-29175 - dataset - line-287] Creating data samplers
[2024-12-31 03:20:59,974 - INFO - pid-29175 - main - line-195] initilizing model with random weights
[2024-12-31 03:21:00,316 - INFO - pid-29176 - dataset - line-287] Creating data samplers
[2024-12-31 03:21:00,319 - INFO - pid-29176 - main - line-195] initilizing model with random weights
[2024-12-31 03:21:00,326 - INFO - pid-29177 - dataset - line-287] Creating data samplers
[2024-12-31 03:21:00,330 - INFO - pid-29177 - main - line-195] initilizing model with random weights
[2024-12-31 03:21:00,334 - INFO - pid-29178 - dataset - line-287] Creating data samplers
[2024-12-31 03:21:00,337 - INFO - pid-29178 - main - line-195] initilizing model with random weights
[2024-12-31 03:21:00,892 - INFO - pid-29177 - main - line-211] Initiate training..
[2024-12-31 03:21:00,892 - INFO - pid-29178 - main - line-211] Initiate training..
[2024-12-31 03:21:00,892 - INFO - pid-29176 - main - line-211] Initiate training..
[2024-12-31 03:21:00,892 - INFO - pid-29175 - main - line-211] Initiate training..
Training [E= 01 | L= 16.857890]: 100%|██████████████████████████████| 626/626 [10:16<00:00,  1.02it/s]
[2024-12-31 03:31:17,381 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 01 | Acc-1: 0.2474 | Acc-5: 1.0059 | Loss: 6.904975
Validation [E= 01 | L= 0.770278]: 100%|██████████████████████████████| 25/25 [01:20<00:00,  3.23s/it]
[2024-12-31 03:32:38,210 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 01 | Acc-1: 0.7361 | Acc-5: 2.2796 | Loss: 6.786208
[2024-12-31 03:32:38,215 - INFO - pid-29175 - torch_utils - line-350] Saving model at 1
Training [E= 02 | L= 16.595882]: 100%|██████████████████████████████| 626/626 [10:11<00:00,  1.02it/s]
[2024-12-31 03:42:50,044 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 02 | Acc-1: 0.4343 | Acc-5: 1.9026 | Loss: 6.797499
Validation [E= 02 | L= 0.775340]: 100%|██████████████████████████████| 25/25 [01:10<00:00,  2.83s/it]
[2024-12-31 03:44:00,766 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 02 | Acc-1: 1.3787 | Acc-5: 6.0553 | Loss: 6.843213
Training [E= 03 | L= 15.575318]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 03:54:09,756 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 03 | Acc-1: 1.9357 | Acc-5: 6.9442 | Loss: 6.379700
Validation [E= 03 | L= 0.663927]: 100%|██████████████████████████████| 25/25 [01:14<00:00,  2.96s/it]
[2024-12-31 03:55:23,822 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 03 | Acc-1: 5.2267 | Acc-5: 16.9086 | Loss: 5.833736
[2024-12-31 03:55:23,824 - INFO - pid-29175 - torch_utils - line-350] Saving model at 3
Training [E= 04 | L= 14.754466]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 04:05:34,344 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 04 | Acc-1: 4.3696 | Acc-5: 13.3674 | Loss: 6.042829
Validation [E= 04 | L= 0.615919]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.92s/it]
[2024-12-31 04:06:47,252 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 04 | Acc-1: 8.4540 | Acc-5: 22.9263 | Loss: 5.424192
[2024-12-31 04:06:47,255 - INFO - pid-29175 - torch_utils - line-350] Saving model at 4
Training [E= 05 | L= 14.081366]: 100%|██████████████████████████████| 626/626 [10:11<00:00,  1.02it/s]
[2024-12-31 04:16:59,849 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 05 | Acc-1: 7.5223 | Acc-5: 19.8735 | Loss: 5.766627
Validation [E= 05 | L= 0.571653]: 100%|██████████████████████████████| 25/25 [01:10<00:00,  2.83s/it]
[2024-12-31 04:18:10,602 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 05 | Acc-1: 14.3878 | Acc-5: 32.1843 | Loss: 5.026327
[2024-12-31 04:18:10,604 - INFO - pid-29175 - torch_utils - line-350] Saving model at 5
Training [E= 06 | L= 13.358067]: 100%|██████████████████████████████| 626/626 [10:11<00:00,  1.02it/s]
[2024-12-31 04:28:22,797 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 06 | Acc-1: 11.4220 | Acc-5: 27.0854 | Loss: 5.472540
Validation [E= 06 | L= 0.512840]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.95s/it]
[2024-12-31 04:29:36,670 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 06 | Acc-1: 20.5538 | Acc-5: 43.2466 | Loss: 4.515450
[2024-12-31 04:29:36,672 - INFO - pid-29175 - torch_utils - line-350] Saving model at 6
Training [E= 07 | L= 12.675102]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 04:39:47,828 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 07 | Acc-1: 15.6980 | Acc-5: 34.0455 | Loss: 5.190147
Validation [E= 07 | L= 0.488466]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.92s/it]
[2024-12-31 04:41:00,894 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 07 | Acc-1: 25.4791 | Acc-5: 50.0307 | Loss: 4.267109
[2024-12-31 04:41:00,896 - INFO - pid-29175 - torch_utils - line-350] Saving model at 7
Training [E= 08 | L= 12.086413]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 04:51:11,631 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 08 | Acc-1: 19.7740 | Acc-5: 39.8397 | Loss: 4.949385
Validation [E= 08 | L= 0.463074]: 100%|██████████████████████████████| 25/25 [01:11<00:00,  2.85s/it]
[2024-12-31 04:52:22,841 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 08 | Acc-1: 29.7670 | Acc-5: 54.6179 | Loss: 4.058676
[2024-12-31 04:52:22,843 - INFO - pid-29175 - torch_utils - line-350] Saving model at 8
Training [E= 09 | L= 11.712688]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.02it/s]
[2024-12-31 05:02:34,580 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 09 | Acc-1: 22.8063 | Acc-5: 43.8998 | Loss: 4.796194
Validation [E= 09 | L= 0.451514]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.92s/it]
[2024-12-31 05:03:47,552 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 09 | Acc-1: 31.4726 | Acc-5: 57.0600 | Loss: 3.955895
[2024-12-31 05:03:47,554 - INFO - pid-29175 - torch_utils - line-350] Saving model at 9
Training [E= 10 | L= 11.275340]: 100%|██████████████████████████████| 626/626 [10:11<00:00,  1.02it/s]
[2024-12-31 05:13:59,463 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 10 | Acc-1: 26.0452 | Acc-5: 47.9608 | Loss: 4.617301
Validation [E= 10 | L= 0.413176]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.96s/it]
[2024-12-31 05:15:13,397 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 10 | Acc-1: 38.3312 | Acc-5: 64.6302 | Loss: 3.623783
[2024-12-31 05:15:13,399 - INFO - pid-29175 - torch_utils - line-350] Saving model at 10
Training [E= 11 | L= 11.114424]: 100%|██████████████████████████████| 626/626 [10:11<00:00,  1.02it/s]
[2024-12-31 05:25:25,323 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 11 | Acc-1: 28.0018 | Acc-5: 50.1594 | Loss: 4.551856
Validation [E= 11 | L= 0.400806]: 100%|██████████████████████████████| 25/25 [01:11<00:00,  2.87s/it]
[2024-12-31 05:26:37,083 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 11 | Acc-1: 41.3187 | Acc-5: 68.1144 | Loss: 3.504982
[2024-12-31 05:26:37,085 - INFO - pid-29175 - torch_utils - line-350] Saving model at 11
Training [E= 12 | L= 10.649436]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 05:36:47,207 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 12 | Acc-1: 30.6641 | Acc-5: 53.5556 | Loss: 4.362741
Validation [E= 12 | L= 0.382671]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.90s/it]
[2024-12-31 05:37:59,759 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 12 | Acc-1: 44.2251 | Acc-5: 70.4661 | Loss: 3.362424
[2024-12-31 05:37:59,762 - INFO - pid-29175 - torch_utils - line-350] Saving model at 12
Training [E= 13 | L= 10.600598]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 05:48:09,726 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 13 | Acc-1: 31.7682 | Acc-5: 54.4850 | Loss: 4.340725
Validation [E= 13 | L= 0.372126]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.93s/it]
[2024-12-31 05:49:23,026 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 13 | Acc-1: 45.4318 | Acc-5: 71.9763 | Loss: 3.253295
[2024-12-31 05:49:23,028 - INFO - pid-29175 - torch_utils - line-350] Saving model at 13
Training [E= 14 | L= 10.466312]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 05:59:34,214 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 14 | Acc-1: 33.1719 | Acc-5: 55.9910 | Loss: 4.285490
Validation [E= 14 | L= 0.361679]: 100%|██████████████████████████████| 25/25 [01:11<00:00,  2.87s/it]
[2024-12-31 06:00:45,867 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 14 | Acc-1: 47.8029 | Acc-5: 74.0940 | Loss: 3.159051
[2024-12-31 06:00:45,898 - INFO - pid-29175 - torch_utils - line-350] Saving model at 14
Training [E= 15 | L= 10.307777]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 06:10:57,259 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 15 | Acc-1: 34.3185 | Acc-5: 57.3585 | Loss: 4.221204
Validation [E= 15 | L= 0.351029]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.92s/it]
[2024-12-31 06:12:10,386 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 15 | Acc-1: 49.3883 | Acc-5: 74.9972 | Loss: 3.076335
[2024-12-31 06:12:10,388 - INFO - pid-29175 - torch_utils - line-350] Saving model at 15
Training [E= 16 | L= 10.110045]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 06:22:21,636 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 16 | Acc-1: 35.7466 | Acc-5: 58.9142 | Loss: 4.143098
Validation [E= 16 | L= 0.344453]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.91s/it]
[2024-12-31 06:23:34,345 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 16 | Acc-1: 50.0699 | Acc-5: 76.1338 | Loss: 3.016102
[2024-12-31 06:23:34,347 - INFO - pid-29175 - torch_utils - line-350] Saving model at 16
Training [E= 17 | L= 10.066474]: 100%|██████████████████████████████| 626/626 [10:06<00:00,  1.03it/s]
[2024-12-31 06:33:42,181 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 17 | Acc-1: 36.1637 | Acc-5: 59.3968 | Loss: 4.124437
Validation [E= 17 | L= 0.359989]: 100%|██████████████████████████████| 25/25 [01:10<00:00,  2.83s/it]
[2024-12-31 06:34:52,832 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 17 | Acc-1: 48.2087 | Acc-5: 74.7322 | Loss: 3.156575
Training [E= 18 | L= 10.049165]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 06:45:02,475 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 18 | Acc-1: 36.9852 | Acc-5: 59.9544 | Loss: 4.117311
Validation [E= 18 | L= 0.346491]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.96s/it]
[2024-12-31 06:46:16,356 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 18 | Acc-1: 51.2507 | Acc-5: 77.5895 | Loss: 3.028115
Training [E= 19 | L= 9.964691]: 100%|██████████████████████████████| 626/626 [10:11<00:00,  1.02it/s]
[2024-12-31 06:56:27,714 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 19 | Acc-1: 37.6551 | Acc-5: 60.5977 | Loss: 4.080768
Validation [E= 19 | L= 0.347531]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.92s/it]
[2024-12-31 06:57:40,621 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 19 | Acc-1: 52.3849 | Acc-5: 77.5056 | Loss: 3.044664
Training [E= 20 | L= 9.779704]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 07:07:49,032 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 20 | Acc-1: 38.6922 | Acc-5: 61.8592 | Loss: 4.005047
Validation [E= 20 | L= 0.329882]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.91s/it]
[2024-12-31 07:09:01,843 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 20 | Acc-1: 54.4232 | Acc-5: 79.3342 | Loss: 2.884025
[2024-12-31 07:09:01,846 - INFO - pid-29175 - torch_utils - line-350] Saving model at 20
Training [E= 21 | L= 9.591530]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.02it/s]
[2024-12-31 07:19:13,492 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 21 | Acc-1: 40.2575 | Acc-5: 63.6313 | Loss: 3.928348
Validation [E= 21 | L= 0.317045]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.93s/it]
[2024-12-31 07:20:26,732 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 21 | Acc-1: 56.4720 | Acc-5: 81.3511 | Loss: 2.771243
[2024-12-31 07:20:26,734 - INFO - pid-29175 - torch_utils - line-350] Saving model at 21
Training [E= 22 | L= 9.435255]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 07:30:36,078 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 22 | Acc-1: 41.1398 | Acc-5: 64.3533 | Loss: 3.863607
Validation [E= 22 | L= 0.341988]: 100%|██████████████████████████████| 25/25 [01:11<00:00,  2.85s/it]
[2024-12-31 07:31:47,340 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 22 | Acc-1: 52.3746 | Acc-5: 77.5102 | Loss: 2.997467
Training [E= 23 | L= 9.438316]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 07:41:56,141 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 23 | Acc-1: 41.5114 | Acc-5: 64.7558 | Loss: 3.867324
Validation [E= 23 | L= 0.321625]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.92s/it]
[2024-12-31 07:43:09,086 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 23 | Acc-1: 56.4552 | Acc-5: 80.9801 | Loss: 2.814265
Training [E= 24 | L= 9.330466]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 07:53:18,005 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 24 | Acc-1: 41.9981 | Acc-5: 65.0693 | Loss: 3.820386
Validation [E= 24 | L= 0.321130]: 100%|██████████████████████████████| 25/25 [01:11<00:00,  2.88s/it]
[2024-12-31 07:54:29,975 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 24 | Acc-1: 56.0631 | Acc-5: 80.9500 | Loss: 2.805141
Training [E= 25 | L= 9.318251]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 08:04:40,105 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 25 | Acc-1: 42.5703 | Acc-5: 65.7442 | Loss: 3.817302
Validation [E= 25 | L= 0.318088]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.91s/it]
[2024-12-31 08:05:52,896 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 25 | Acc-1: 56.4259 | Acc-5: 81.0758 | Loss: 2.784122
Training [E= 26 | L= 9.263417]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 08:16:02,033 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 26 | Acc-1: 43.2077 | Acc-5: 66.2830 | Loss: 3.795581
Validation [E= 26 | L= 0.331219]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.90s/it]
[2024-12-31 08:17:14,606 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 26 | Acc-1: 56.7989 | Acc-5: 81.0817 | Loss: 2.905150
Training [E= 27 | L= 9.168669]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 08:27:24,951 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 27 | Acc-1: 43.6099 | Acc-5: 66.6069 | Loss: 3.754170
Validation [E= 27 | L= 0.306421]: 100%|██████████████████████████████| 25/25 [01:11<00:00,  2.84s/it]
[2024-12-31 08:28:35,959 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 27 | Acc-1: 59.4505 | Acc-5: 84.0144 | Loss: 2.677052
[2024-12-31 08:28:35,964 - INFO - pid-29175 - torch_utils - line-350] Saving model at 27
Training [E= 28 | L= 9.145703]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 08:38:46,933 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 28 | Acc-1: 43.7428 | Acc-5: 66.7332 | Loss: 3.747097
Validation [E= 28 | L= 0.311970]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.93s/it]
[2024-12-31 08:40:00,222 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 28 | Acc-1: 59.3451 | Acc-5: 82.8323 | Loss: 2.734971
Training [E= 29 | L= 9.018493]: 100%|██████████████████████████████| 626/626 [10:06<00:00,  1.03it/s]
[2024-12-31 08:50:06,592 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 29 | Acc-1: 44.7253 | Acc-5: 67.6583 | Loss: 3.693665
Validation [E= 29 | L= 0.302260]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.93s/it]
[2024-12-31 08:51:19,818 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 29 | Acc-1: 60.0078 | Acc-5: 83.6232 | Loss: 2.645771
[2024-12-31 08:51:19,820 - INFO - pid-29175 - torch_utils - line-350] Saving model at 29
Training [E= 30 | L= 9.070536]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 09:01:31,140 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 30 | Acc-1: 44.8289 | Acc-5: 67.6805 | Loss: 3.713694
Validation [E= 30 | L= 0.301907]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.90s/it]
[2024-12-31 09:02:43,654 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 30 | Acc-1: 62.2115 | Acc-5: 84.8048 | Loss: 2.636492
[2024-12-31 09:02:43,656 - INFO - pid-29175 - torch_utils - line-350] Saving model at 30
Training [E= 31 | L= 9.057574]: 100%|██████████████████████████████| 626/626 [10:12<00:00,  1.02it/s]
[2024-12-31 09:12:57,064 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 31 | Acc-1: 44.8941 | Acc-5: 67.7781 | Loss: 3.711755
Validation [E= 31 | L= 0.324808]: 100%|██████████████████████████████| 25/25 [01:11<00:00,  2.84s/it]
[2024-12-31 09:14:08,115 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 31 | Acc-1: 58.6543 | Acc-5: 81.7522 | Loss: 2.845606
Training [E= 32 | L= 8.943161]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 09:24:16,324 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 32 | Acc-1: 45.9811 | Acc-5: 68.9556 | Loss: 3.661735
Validation [E= 32 | L= 0.328917]: 100%|██████████████████████████████| 25/25 [01:11<00:00,  2.87s/it]
[2024-12-31 09:25:28,130 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 32 | Acc-1: 58.8516 | Acc-5: 81.6571 | Loss: 2.880881
Training [E= 33 | L= 8.950845]: 100%|██████████████████████████████| 626/626 [10:07<00:00,  1.03it/s]
[2024-12-31 09:35:36,024 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 33 | Acc-1: 46.0129 | Acc-5: 68.8395 | Loss: 3.665713
Validation [E= 33 | L= 0.293185]: 100%|██████████████████████████████| 25/25 [01:11<00:00,  2.87s/it]
[2024-12-31 09:36:47,784 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 33 | Acc-1: 62.0097 | Acc-5: 85.0828 | Loss: 2.567299
[2024-12-31 09:36:47,786 - INFO - pid-29175 - torch_utils - line-350] Saving model at 33
Training [E= 34 | L= 8.843836]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 09:46:57,740 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 34 | Acc-1: 46.3321 | Acc-5: 69.0907 | Loss: 3.624871
Validation [E= 34 | L= 0.291347]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.90s/it]
[2024-12-31 09:48:10,156 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 34 | Acc-1: 63.9758 | Acc-5: 86.2599 | Loss: 2.549229
[2024-12-31 09:48:10,158 - INFO - pid-29175 - torch_utils - line-350] Saving model at 34
Training [E= 35 | L= 8.821557]: 100%|██████████████████████████████| 626/626 [10:11<00:00,  1.02it/s]
[2024-12-31 09:58:22,282 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 35 | Acc-1: 46.4834 | Acc-5: 69.3035 | Loss: 3.612334
Validation [E= 35 | L= 0.288820]: 100%|██████████████████████████████| 25/25 [01:11<00:00,  2.87s/it]
[2024-12-31 09:59:34,094 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 35 | Acc-1: 63.0475 | Acc-5: 85.6831 | Loss: 2.528886
[2024-12-31 09:59:34,097 - INFO - pid-29175 - torch_utils - line-350] Saving model at 35
Training [E= 36 | L= 8.834569]: 100%|██████████████████████████████| 626/626 [10:11<00:00,  1.02it/s]
[2024-12-31 10:09:46,854 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 36 | Acc-1: 46.8756 | Acc-5: 69.6782 | Loss: 3.617456
Validation [E= 36 | L= 0.346586]: 100%|██████████████████████████████| 25/25 [01:11<00:00,  2.85s/it]
[2024-12-31 10:10:58,126 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 36 | Acc-1: 61.8184 | Acc-5: 83.7313 | Loss: 3.034966
Training [E= 37 | L= 8.595072]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 10:21:08,031 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 37 | Acc-1: 48.4038 | Acc-5: 71.1908 | Loss: 3.519996
Validation [E= 37 | L= 0.298560]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.88s/it]
[2024-12-31 10:22:20,055 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 37 | Acc-1: 61.7557 | Acc-5: 85.0991 | Loss: 2.611492
Training [E= 38 | L= 8.718984]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.02it/s]
[2024-12-31 10:32:30,931 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 38 | Acc-1: 47.1951 | Acc-5: 69.9943 | Loss: 3.569921
Validation [E= 38 | L= 0.299695]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.89s/it]
[2024-12-31 10:33:43,126 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 38 | Acc-1: 62.5650 | Acc-5: 85.2437 | Loss: 2.620241
Training [E= 39 | L= 8.609993]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 10:43:52,436 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 39 | Acc-1: 48.3935 | Acc-5: 71.0463 | Loss: 3.526161
Validation [E= 39 | L= 0.287857]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.92s/it]
[2024-12-31 10:45:05,353 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 39 | Acc-1: 64.8339 | Acc-5: 86.7599 | Loss: 2.518393
[2024-12-31 10:45:05,355 - INFO - pid-29175 - torch_utils - line-350] Saving model at 39
Training [E= 40 | L= 8.655353]: 100%|██████████████████████████████| 626/626 [10:05<00:00,  1.03it/s]
[2024-12-31 10:55:11,984 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 40 | Acc-1: 48.2425 | Acc-5: 71.0273 | Loss: 3.545894
Validation [E= 40 | L= 0.302006]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.94s/it]
[2024-12-31 10:56:25,527 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 40 | Acc-1: 62.1921 | Acc-5: 84.6604 | Loss: 2.648105
Training [E= 41 | L= 8.587264]: 100%|██████████████████████████████| 626/626 [10:06<00:00,  1.03it/s]
[2024-12-31 11:06:32,061 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 41 | Acc-1: 49.0774 | Acc-5: 71.6915 | Loss: 3.515885
Validation [E= 41 | L= 0.290464]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.89s/it]
[2024-12-31 11:07:44,330 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 41 | Acc-1: 64.3404 | Acc-5: 86.2437 | Loss: 2.545570
Training [E= 42 | L= 8.626280]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 11:17:53,980 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 42 | Acc-1: 48.3779 | Acc-5: 71.0835 | Loss: 3.534871
Validation [E= 42 | L= 0.295010]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.89s/it]
[2024-12-31 11:19:06,351 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 42 | Acc-1: 63.8906 | Acc-5: 86.3843 | Loss: 2.584526
Training [E= 43 | L= 8.543396]: 100%|██████████████████████████████| 626/626 [10:06<00:00,  1.03it/s]
[2024-12-31 11:29:12,576 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 43 | Acc-1: 48.9018 | Acc-5: 71.5290 | Loss: 3.500790
Validation [E= 43 | L= 0.302655]: 100%|██████████████████████████████| 25/25 [01:11<00:00,  2.86s/it]
[2024-12-31 11:30:24,099 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 43 | Acc-1: 62.8333 | Acc-5: 85.6408 | Loss: 2.652227
Training [E= 44 | L= 8.575325]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 11:40:33,904 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 44 | Acc-1: 48.8596 | Acc-5: 71.4357 | Loss: 3.511448
Validation [E= 44 | L= 0.283135]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.93s/it]
[2024-12-31 11:41:47,108 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 44 | Acc-1: 66.1477 | Acc-5: 87.5861 | Loss: 2.478016
[2024-12-31 11:41:47,110 - INFO - pid-29175 - torch_utils - line-350] Saving model at 44
Training [E= 45 | L= 8.325772]: 100%|██████████████████████████████| 626/626 [10:11<00:00,  1.02it/s]
[2024-12-31 11:51:59,055 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 45 | Acc-1: 50.4755 | Acc-5: 72.9798 | Loss: 3.408986
Validation [E= 45 | L= 0.280843]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.88s/it]
[2024-12-31 11:53:11,060 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 45 | Acc-1: 65.1601 | Acc-5: 87.1408 | Loss: 2.460171
[2024-12-31 11:53:11,062 - INFO - pid-29175 - torch_utils - line-350] Saving model at 45
Training [E= 46 | L= 8.488901]: 100%|██████████████████████████████| 626/626 [10:07<00:00,  1.03it/s]
[2024-12-31 12:03:19,117 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 46 | Acc-1: 49.9158 | Acc-5: 72.3137 | Loss: 3.476239
Validation [E= 46 | L= 0.284571]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.90s/it]
[2024-12-31 12:04:31,611 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 46 | Acc-1: 64.8104 | Acc-5: 87.2033 | Loss: 2.490840
Training [E= 47 | L= 8.409434]: 100%|██████████████████████████████| 626/626 [10:07<00:00,  1.03it/s]
[2024-12-31 12:14:39,404 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 47 | Acc-1: 50.2998 | Acc-5: 72.6953 | Loss: 3.443290
Validation [E= 47 | L= 0.291130]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.93s/it]
[2024-12-31 12:15:52,684 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 47 | Acc-1: 65.3632 | Acc-5: 86.9982 | Loss: 2.549323
Training [E= 48 | L= 8.349556]: 100%|██████████████████████████████| 626/626 [10:05<00:00,  1.03it/s]
[2024-12-31 12:25:58,469 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 48 | Acc-1: 50.6926 | Acc-5: 73.0559 | Loss: 3.421440
Validation [E= 48 | L= 0.284883]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.93s/it]
[2024-12-31 12:27:11,732 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 48 | Acc-1: 64.9979 | Acc-5: 87.0672 | Loss: 2.493191
Training [E= 49 | L= 8.399290]: 100%|██████████████████████████████| 626/626 [10:07<00:00,  1.03it/s]
[2024-12-31 12:37:19,529 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 49 | Acc-1: 50.4315 | Acc-5: 72.8282 | Loss: 3.440680
Validation [E= 49 | L= 0.276758]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.93s/it]
[2024-12-31 12:38:32,694 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 49 | Acc-1: 66.7588 | Acc-5: 88.0053 | Loss: 2.416991
[2024-12-31 12:38:32,696 - INFO - pid-29175 - torch_utils - line-350] Saving model at 49
Training [E= 50 | L= 8.217108]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 12:48:42,627 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 50 | Acc-1: 51.7591 | Acc-5: 74.0549 | Loss: 3.364445
Validation [E= 50 | L= 0.291582]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.94s/it]
[2024-12-31 12:49:56,097 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 50 | Acc-1: 65.9705 | Acc-5: 87.0223 | Loss: 2.547282
Training [E= 51 | L= 8.275798]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 13:00:04,172 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 51 | Acc-1: 51.2645 | Acc-5: 73.6506 | Loss: 3.388606
Validation [E= 51 | L= 0.282936]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.89s/it]
[2024-12-31 13:01:16,532 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 51 | Acc-1: 66.6801 | Acc-5: 88.0489 | Loss: 2.473979
Training [E= 52 | L= 8.183124]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 13:11:24,549 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 52 | Acc-1: 51.8255 | Acc-5: 74.0565 | Loss: 3.350389
Validation [E= 52 | L= 0.272516]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.93s/it]
[2024-12-31 13:12:37,839 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 52 | Acc-1: 67.6176 | Acc-5: 88.3757 | Loss: 2.381323
[2024-12-31 13:12:37,841 - INFO - pid-29175 - torch_utils - line-350] Saving model at 52
Training [E= 53 | L= 8.149978]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 13:22:47,041 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 53 | Acc-1: 52.5550 | Acc-5: 74.6044 | Loss: 3.337284
Validation [E= 53 | L= 0.279062]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.92s/it]
[2024-12-31 13:24:00,163 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 53 | Acc-1: 66.5494 | Acc-5: 87.4923 | Loss: 2.447297
Training [E= 54 | L= 8.178603]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 13:34:10,820 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 54 | Acc-1: 52.1725 | Acc-5: 74.4334 | Loss: 3.352643
Validation [E= 54 | L= 0.278332]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.92s/it]
[2024-12-31 13:35:23,798 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 54 | Acc-1: 68.5734 | Acc-5: 88.9239 | Loss: 2.439190
Training [E= 55 | L= 8.202357]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 13:45:31,974 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 55 | Acc-1: 52.1924 | Acc-5: 74.2016 | Loss: 3.358259
Validation [E= 55 | L= 0.270872]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.89s/it]
[2024-12-31 13:46:44,189 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 55 | Acc-1: 69.0578 | Acc-5: 89.3868 | Loss: 2.369739
[2024-12-31 13:46:44,193 - INFO - pid-29175 - torch_utils - line-350] Saving model at 55
Training [E= 56 | L= 8.059036]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 13:56:54,943 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 56 | Acc-1: 53.1843 | Acc-5: 75.0986 | Loss: 3.302475
Validation [E= 56 | L= 0.276065]: 100%|██████████████████████████████| 25/25 [01:10<00:00,  2.84s/it]
[2024-12-31 13:58:05,824 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 56 | Acc-1: 67.5629 | Acc-5: 88.5691 | Loss: 2.415706
Training [E= 57 | L= 8.019959]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 14:08:14,626 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 57 | Acc-1: 53.5259 | Acc-5: 75.3841 | Loss: 3.284116
Validation [E= 57 | L= 0.279179]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.90s/it]
[2024-12-31 14:09:27,245 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 57 | Acc-1: 66.5357 | Acc-5: 87.6050 | Loss: 2.447679
Training [E= 58 | L= 7.986930]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 14:19:35,332 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 58 | Acc-1: 53.5777 | Acc-5: 75.4012 | Loss: 3.272371
Validation [E= 58 | L= 0.267822]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.92s/it]
[2024-12-31 14:20:48,425 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 58 | Acc-1: 69.5161 | Acc-5: 89.4337 | Loss: 2.340624
[2024-12-31 14:20:48,427 - INFO - pid-29175 - torch_utils - line-350] Saving model at 58
Training [E= 59 | L= 7.986249]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 14:30:58,981 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 59 | Acc-1: 53.3896 | Acc-5: 75.4883 | Loss: 3.272451
Validation [E= 59 | L= 0.267662]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.91s/it]
[2024-12-31 14:32:11,757 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 59 | Acc-1: 69.3741 | Acc-5: 89.8822 | Loss: 2.341684
Training [E= 60 | L= 8.044595]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 14:42:21,326 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 60 | Acc-1: 53.4888 | Acc-5: 75.3391 | Loss: 3.295915
Validation [E= 60 | L= 0.258811]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.89s/it]
[2024-12-31 14:43:33,687 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 60 | Acc-1: 70.9919 | Acc-5: 90.4337 | Loss: 2.264673
[2024-12-31 14:43:33,690 - INFO - pid-29175 - torch_utils - line-350] Saving model at 60
Training [E= 61 | L= 8.036264]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 14:53:44,131 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 61 | Acc-1: 53.3971 | Acc-5: 75.2948 | Loss: 3.293148
Validation [E= 61 | L= 0.277518]: 100%|██████████████████████████████| 25/25 [01:11<00:00,  2.87s/it]
[2024-12-31 14:54:55,984 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 61 | Acc-1: 69.6756 | Acc-5: 89.9525 | Loss: 2.431982
Training [E= 62 | L= 7.968865]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 15:05:06,126 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 62 | Acc-1: 53.8941 | Acc-5: 75.7956 | Loss: 3.265245
Validation [E= 62 | L= 0.262836]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.90s/it]
[2024-12-31 15:06:18,570 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 62 | Acc-1: 69.6971 | Acc-5: 90.2111 | Loss: 2.302200
Training [E= 63 | L= 8.026529]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 15:16:28,815 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 63 | Acc-1: 53.9016 | Acc-5: 75.7313 | Loss: 3.286623
Validation [E= 63 | L= 0.269305]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.90s/it]
[2024-12-31 15:17:41,291 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 63 | Acc-1: 68.7095 | Acc-5: 89.3087 | Loss: 2.356936
Training [E= 64 | L= 7.904493]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 15:27:51,805 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 64 | Acc-1: 54.5477 | Acc-5: 76.2695 | Loss: 3.236771
Validation [E= 64 | L= 0.262069]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.95s/it]
[2024-12-31 15:29:05,500 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 64 | Acc-1: 71.2731 | Acc-5: 90.8555 | Loss: 2.294588
Training [E= 65 | L= 7.776047]: 100%|██████████████████████████████| 626/626 [10:11<00:00,  1.02it/s]
[2024-12-31 15:39:16,590 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 65 | Acc-1: 55.3302 | Acc-5: 76.9441 | Loss: 3.186902
Validation [E= 65 | L= 0.262108]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.91s/it]
[2024-12-31 15:40:29,314 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 65 | Acc-1: 70.7224 | Acc-5: 90.6882 | Loss: 2.292482
Training [E= 66 | L= 7.686599]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.02it/s]
[2024-12-31 15:50:40,314 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 66 | Acc-1: 56.2147 | Acc-5: 77.6860 | Loss: 3.151156
Validation [E= 66 | L= 0.270020]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.90s/it]
[2024-12-31 15:51:52,931 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 66 | Acc-1: 71.6619 | Acc-5: 91.0131 | Loss: 2.366183
Training [E= 67 | L= 7.578947]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 16:02:01,562 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 67 | Acc-1: 56.4909 | Acc-5: 78.0030 | Loss: 3.105660
Validation [E= 67 | L= 0.261287]: 100%|██████████████████████████████| 25/25 [01:10<00:00,  2.82s/it]
[2024-12-31 16:03:12,065 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 67 | Acc-1: 71.5473 | Acc-5: 90.8725 | Loss: 2.291297
Training [E= 68 | L= 7.876544]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 16:13:21,213 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 68 | Acc-1: 55.2681 | Acc-5: 76.7778 | Loss: 3.227403
Validation [E= 68 | L= 0.263820]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.91s/it]
[2024-12-31 16:14:34,031 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 68 | Acc-1: 71.8025 | Acc-5: 91.0444 | Loss: 2.309612
Training [E= 69 | L= 7.738898]: 100%|██████████████████████████████| 626/626 [10:09<00:00,  1.03it/s]
[2024-12-31 16:24:43,387 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 69 | Acc-1: 56.1676 | Acc-5: 77.5203 | Loss: 3.169855
Validation [E= 69 | L= 0.254693]: 100%|██████████████████████████████| 25/25 [01:10<00:00,  2.83s/it]
[2024-12-31 16:25:54,247 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 69 | Acc-1: 72.2615 | Acc-5: 91.0476 | Loss: 2.231605
[2024-12-31 16:25:54,250 - INFO - pid-29175 - torch_utils - line-350] Saving model at 69
Training [E= 70 | L= 7.767695]: 100%|██████████████████████████████| 626/626 [10:07<00:00,  1.03it/s]
[2024-12-31 16:36:02,923 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 70 | Acc-1: 55.9283 | Acc-5: 77.4233 | Loss: 3.182815
Validation [E= 70 | L= 0.260290]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.90s/it]
[2024-12-31 16:37:15,444 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 70 | Acc-1: 72.0727 | Acc-5: 91.2103 | Loss: 2.279680
Training [E= 71 | L= 7.735346]: 100%|██████████████████████████████| 626/626 [10:07<00:00,  1.03it/s]
[2024-12-31 16:47:22,916 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 71 | Acc-1: 55.9573 | Acc-5: 77.4427 | Loss: 3.167196
Validation [E= 71 | L= 0.245994]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.94s/it]
[2024-12-31 16:48:36,413 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 71 | Acc-1: 72.7575 | Acc-5: 91.5430 | Loss: 2.153832
[2024-12-31 16:48:36,415 - INFO - pid-29175 - torch_utils - line-350] Saving model at 71
Training [E= 72 | L= 7.818854]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 16:58:47,733 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 72 | Acc-1: 56.3701 | Acc-5: 77.5631 | Loss: 3.201098
Validation [E= 72 | L= 0.257260]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.91s/it]
[2024-12-31 17:00:00,511 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 72 | Acc-1: 72.1247 | Acc-5: 91.4792 | Loss: 2.251054
Training [E= 73 | L= 7.584355]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 17:10:09,072 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 73 | Acc-1: 57.2784 | Acc-5: 78.4866 | Loss: 3.106045
Validation [E= 73 | L= 0.262350]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.93s/it]
[2024-12-31 17:11:22,412 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 73 | Acc-1: 73.4496 | Acc-5: 91.8432 | Loss: 2.300553
Training [E= 74 | L= 7.566512]: 100%|██████████████████████████████| 626/626 [10:10<00:00,  1.03it/s]
[2024-12-31 17:21:33,050 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 74 | Acc-1: 57.1542 | Acc-5: 78.5081 | Loss: 3.098093
Validation [E= 74 | L= 0.254289]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.91s/it]
[2024-12-31 17:22:45,804 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 74 | Acc-1: 73.3076 | Acc-5: 91.8900 | Loss: 2.224742
Training [E= 75 | L= 7.582364]: 100%|██████████████████████████████| 626/626 [10:06<00:00,  1.03it/s]
[2024-12-31 17:32:52,623 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 75 | Acc-1: 57.5635 | Acc-5: 78.4348 | Loss: 3.106746
Validation [E= 75 | L= 0.239476]: 100%|██████████████████████████████| 25/25 [01:12<00:00,  2.90s/it]
[2024-12-31 17:34:05,158 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 75 | Acc-1: 74.6377 | Acc-5: 92.5964 | Loss: 2.096875
[2024-12-31 17:34:05,160 - INFO - pid-29175 - torch_utils - line-350] Saving model at 75
Training [E= 76 | L= 7.503713]: 100%|██████████████████████████████| 626/626 [10:08<00:00,  1.03it/s]
[2024-12-31 17:44:14,433 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 76 | Acc-1: 58.1535 | Acc-5: 79.0357 | Loss: 3.074724
Validation [E= 76 | L= 0.241554]: 100%|██████████████████████████████| 25/25 [01:14<00:00,  2.96s/it]
[2024-12-31 17:45:28,512 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 76 | Acc-1: 74.3448 | Acc-5: 92.4851 | Loss: 2.117143
Training [E= 77 | L= 7.597540]: 100%|██████████████████████████████| 626/626 [10:05<00:00,  1.03it/s]
[2024-12-31 17:55:34,140 - INFO - pid-29175 - torch_utils - line-482] [Train] E: 77 | Acc-1: 57.8100 | Acc-5: 78.8398 | Loss: 3.110456
Validation [E= 77 | L= 0.247096]: 100%|██████████████████████████████| 25/25 [01:13<00:00,  2.94s/it]
[2024-12-31 17:56:47,543 - INFO - pid-29175 - torch_utils - line-527] [Valid] Epoch: 77 | Acc-1: 74.6794 | Acc-5: 92.4760 | Loss: 2.163911
```


# AWS TRAINING LOGS SCREEN-SHOT

![Training logs](<assets/Training logs on aws.png>)


# TRAINING CURVE

## ACCURACY-LOSS

![accuracy-loss-curve](<assets/accuracy-loss-curve.png>)


## LEARNING RATE

![alt text](<assets/lr curve.png>)


# RESULT

| **Accuracy**  |  **Top-1**  |  **Top-5**  |  **Epoch**  |
|---------------|-------------|-------------|-------------|
|     Test      |   74.63%    |    92.59%   |      75     |


# CITETION

```bibtex
@misc{imagenet-object-localization-challenge,
    author = {Addison Howard and Eunbyung Park and Wendy Kan},
    title = {ImageNet Object Localization Challenge},
    year = {2018},
    howpublished = {\url{https://kaggle.com/competitions/imagenet-object-localization-challenge}},
    note = {Kaggle}
}
@misc{he2015deepresiduallearningimage,
      title={Deep Residual Learning for Image Recognition}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1512.03385}, 
}
```