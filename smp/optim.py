import torch
import ckconv
import math

# typing
from omegaconf import OmegaConf


DATASET_SIZES = {
    "CharTrajectories": 2000,
    "SpeechCommands": 24482,
    "MNIST": 60000,
    "sMNIST": 60000,
    "CIFAR10": 50000,
    "sCIFAR10": 50000,
    "CIFAR100": 50000,
}


def construct_optimizer(
    model: torch.nn.Module,
    cfg: OmegaConf,
):
    """
    Constructs an optimizer for a given model
    :param model: a list of parameters to be trained
    :param cfg:

    :return: optimizer
    """

    # Unpack values from cfg.train
    optimizer_type = cfg.train.optimizer
    lr = cfg.train.lr
    radius_lr_factor = cfg.train.radius_lr_factor

    # Unpack values from cfg.train.optimizer_params
    momentum = cfg.train.optimizer_params.momentum
    nesterov = cfg.train.optimizer_params.nesterov

    others, radius, no_decay = [], [], []
    for name, param in model.named_parameters():
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        elif "weight_coord" in name:
            no_decay.append(param)
        elif "radius" in name:
            radius.append(param)
        else:
            others.append(param)

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            [
                {"params": others},
                {"params": no_decay, "weight_decay": 0.},
                {"params": radius, "weight_decay": 0., "lr": radius_lr_factor * lr}
            ],
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=cfg.train.weight_decay,
        )

    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            [
                {"params": others},
                {"params": no_decay, "weight_decay": 0.},
                {"params": radius, "weight_decay": 0., "lr": radius_lr_factor * lr}
            ],
            lr=lr,
            weight_decay=cfg.train.weight_decay,
        )
    elif optimizer_type == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            # weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(
            f"Unexpected value for type of optimizer (cfg.train.optimizer): {optimizer_type}"
        )

    return optimizer


def construct_scheduler(
    optimizer,
    cfg: OmegaConf,
):
    """
    Creates a learning rate scheduler for a given model
    :param optimizer: the optimizer to be used
    :return: scheduler
    """

    # Unpack values from cfg.train.scheduler_params
    scheduler_type = cfg.train.scheduler
    decay_factor = cfg.train.scheduler_params.decay_factor
    decay_steps = cfg.train.scheduler_params.decay_steps
    patience = cfg.train.scheduler_params.patience
    warmup_epochs = cfg.train.scheduler_params.warmup_epochs
    warmup = warmup_epochs != -1

    if scheduler_type == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=decay_steps,
            gamma=1.0 / decay_factor,
        )
    elif scheduler_type == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=1.0 / decay_factor,
            patience=patience,
            verbose=True,
        )
    elif scheduler_type == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=decay_factor,
            last_epoch=-1,
        )
    elif scheduler_type == "cosine":
        size_dataset = DATASET_SIZES[cfg.dataset]

        if warmup:
            # If warmup is used, then we need to substract this from T_max.
            T_max = (cfg.train.epochs - warmup_epochs) * math.ceil(
                size_dataset / float(cfg.train.batch_size)
            )  # - warmup epochs
        else:
            T_max = cfg.train.epochs * math.ceil(
                size_dataset / float(cfg.train.batch_size)
            )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=1e-6,
        )
    else:
        lr_scheduler = None
        print(
            f"WARNING! No scheduler will be used. cfg.train.scheduler = {scheduler_type}"
        )

    if warmup and lr_scheduler is not None:
        size_dataset = DATASET_SIZES[cfg.dataset]

        lr_scheduler = ckconv.nn.LinearWarmUp_LRScheduler(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            warmup_iterations=warmup_epochs
            * math.ceil(size_dataset / float(cfg.train.batch_size)),
        )

    return lr_scheduler
