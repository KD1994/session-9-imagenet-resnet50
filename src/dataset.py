import os
import time
import math
import logging
import torch
import torch.distributed as dist

from torch import is_distributed
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate


logger = logging.getLogger(__name__)


def get_mixup_cutmix(*, mixup_alpha, cutmix_alpha, num_classes):
    mixup_cutmix = []
    if mixup_alpha > 0:
        mixup_cutmix.append(
            v2.MixUp(alpha=mixup_alpha, num_classes=num_classes)
        )
    if cutmix_alpha > 0:
        mixup_cutmix.append(
            v2.CutMix(alpha=mixup_alpha, num_classes=num_classes)
        )
    if not mixup_cutmix:
        return None

    return v2.RandomChoice(mixup_cutmix)


class CustomImageNet1KDataset(Dataset):
    def __init__(self, dataset, split, transform=None):
        self.data = dataset
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image']
        if image.mode == "L":
            image = image.convert("RGB")
        label = self.data[idx]['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class RASampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class TrainTransform:
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
    ):
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(v2.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms.append(v2.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
        if hflip_prob > 0:
            transforms.append(v2.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                transforms.append(v2.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                transforms.append(v2.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                transforms.append(v2.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = v2.AutoAugmentPolicy(auto_augment_policy)
                transforms.append(v2.AutoAugment(policy=aa_policy, interpolation=interpolation))

        if backend == "pil":
            transforms.append(v2.PILToTensor())

        transforms.extend(
            [
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            transforms.append(v2.RandomErasing(p=random_erase_prob))

        transforms.append(v2.ToPureTensor())
        self.transforms = v2.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class TestTransform:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        backend="pil"
    ):
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(v2.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms += [
            v2.Resize(resize_size, interpolation=interpolation, antialias=True),
            v2.CenterCrop(crop_size),
        ]

        if backend == "pil":
            transforms.append(v2.PILToTensor())

        transforms += [
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]

        transforms.append(v2.ToPureTensor())

        self.transforms = v2.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


# def _get_hf_dataset(data_path):
#     login()
#     logger.info("Downloading & preparing dataset")
#     st = time.time()
#     builder = load_dataset_builder("ILSVRC/imagenet-1k")
#     # , cache_dir=os.path.join(R"/workspace/extra-data-storage/hf_data/cache"))
#     builder.download_and_prepare(os.path.join(data_path))
#     logger.info(f"Took {time.time() - st}")

#     return builder.as_dataset(split="train"), builder.as_dataset(split="validation")


def get_data_loaders(args):

    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode("bilinear")

    # tr_hf_dataset, ts_hf_dataset = _get_hf_dataset(args.data_dir)
    # num_classes = tr_hf_dataset.features['label'].num_classes

    logger.info("load training data")
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = getattr(args, "ra_magnitude", None)
    augmix_severity = getattr(args, "augmix_severity", None)

    # Create the datasets
    train_dataset = ImageFolder(
        os.path.join(args.data_dir, "train"),
        transform=TrainTransform(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
        )
    )

    test_dataset = ImageFolder(
        os.path.join(args.data_dir, "val"), 
        transform=TestTransform(
            crop_size=val_crop_size,
            resize_size=val_resize_size,
            interpolation=interpolation,
        )
    )

    num_classes = len(train_dataset.classes)

    # train_dataset = CustomImageNet1KDataset(
    #     tr_hf_dataset, 
    #     'train', 
    #     transform=TrainTransform(
    #         crop_size=train_crop_size,
    #         interpolation=interpolation,
    #         auto_augment_policy=auto_augment_policy,
    #         random_erase_prob=random_erase_prob,
    #         ra_magnitude=ra_magnitude,
    #         augmix_severity=augmix_severity,
    #     )
    # )
    
    # test_dataset = CustomImageNet1KDataset(
    #     ts_hf_dataset, 
    #     'test', 
    #     transform=TestTransform(
    #         crop_size=val_crop_size,
    #         resize_size=val_resize_size,
    #         interpolation=interpolation,
    #     )
    # )

    logging.info("Creating data samplers")
    if is_distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(train_dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        test_sampler = SequentialSampler(test_dataset)

    collate_fn = None
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, 
        cutmix_alpha=args.cutmix_alpha, 
        num_classes=num_classes
    )

    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))
    else:
        collate_fn = default_collate

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_sz,
        sampler=train_sampler,
        num_workers=args.data_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_sz, 
        sampler=test_sampler, 
        num_workers=args.data_workers, 
        pin_memory=True
    )

    return train_loader, test_loader, num_classes
