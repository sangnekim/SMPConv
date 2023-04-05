import math
import os
from hydra import utils
from torchvision import datasets, transforms


class CIFAR100(datasets.CIFAR100):  # TODO: Documentation
    def __init__(
        self,
        partition: str,
        **kwargs,
    ):
        if "root" in kwargs:
            root = kwargs["root"]
        else:
            root = utils.get_original_cwd()
            # DEBUG
            # root = "../"
            root = os.path.join(root, "data")

        transform = []
 
        augment = kwargs["augment"]
        if augment == "resnet":
            transform.extend(
                augmentations_resnet()
            )
        elif augment == "None":
            transform.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            raise NotImplementedError(f"augment = {augment}")

        transform = transforms.Compose(transform)

        if partition == "train":
            train = True
        elif partition == "test":
            train = False
        else:
            raise NotImplementedError(
                "The dataset partition {} does not exist".format(partition)
            )

        super().__init__(root=root, train=train, transform=transform, download=True)


def augmentations_resnet(crop_size=None):
    """
    Following "A branching and merging convolutional network with homogeneous filter capsules"
    - Biearly et al., 2020 - https://arxiv.org/abs/2001.09136
    """
    if crop_size is None:
        crop_size = 32
    pad_size = crop_size // 8

    augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(crop_size, pad_size),
    ]

    augmentations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    return augmentations
