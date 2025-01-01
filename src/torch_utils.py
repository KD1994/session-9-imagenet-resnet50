"""
Utility methods and classes for PyTorch training and inference for writer recognition tasks.
"""

import os
import sys
import logging
import time
import torch

from typing import List, Optional, Tuple
from torch.nn import CrossEntropyLoss
# from torch.optim.swa_utils import AveragedModel
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR, StepLR, CyclicLR, CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.model import create_model
from src.utils import accuracy


logger = logging.getLogger(__name__)


def ddp_setup():
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def wrapup_ddp():
    destroy_process_group()


def list_optims():
    return ["adamw", "adam", "sgd", "sgd-nesterov"]


def list_lr_schedulers():
    return ["warmup", "step", "cyclic", "cosineannealinglr", "na"]


def list_criterions():
    return ["ce-with-label-smoothing", "ce"]


def get_hyperparams(args):
    return {
        "criterion": args.criterion,
        "optimizer": args.optim,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "lr_scheduler": args.lr_scheduler,
        "lr_warmup_epochs": args.lr_warmup_epochs,
        "lr_warmup_method": args.lr_warmup_method,
        "lr_warmup_decay": args.lr_warmup_decay,
        "weight_decay": args.decay,
        "norm_weight_decay": args.norm_weight_decay,
        "momentum": args.momentum,
        "num_epochs_warmup": args.warmup_epochs,
        "batch_size": args.batch_sz,
        "epochs": args.epochs
    }


def get_optimizer(net_params, hyper_params):
    if hyper_params["optimizer"] == "adamw":
        return torch.optim.AdamW(net_params, 
                                 lr=hyper_params["lr"], 
                                 weight_decay=hyper_params["weight_decay"])
    elif hyper_params["optimizer"] == "adam":
        return torch.optim.Adam(net_params,
                                lr=hyper_params["lr"],
                                weight_decay=hyper_params["weight_decay"])
    elif hyper_params["optimizer"] == "sgd":
        return torch.optim.SGD(net_params,
                               lr=hyper_params["lr"],
                               momentum=hyper_params["momentum"],
                               nesterov="nesterov" in hyper_params["optimizer"])
    else:
        assert False, "Optimizer not implemented"

    
def get_criterion(hyper_params):
    if hyper_params["criterion"] == "ce":
        return CrossEntropyLoss()
    elif hyper_params["criterion"] == "ce-with-label-smoothing":
        return CrossEntropyLoss(label_smoothing=0.1)
    else:
        assert False, "Criterion not implemented"


def get_lr_scheduler(optimizer, hyper_params):
    if hyper_params["lr_scheduler"] == "warmup":
        return WarmUpLR(optimizer=optimizer, 
                        initial_lr=hyper_params["lr"], 
                        num_epochs_warm_up=hyper_params["num_epochs_warmup"])
    elif hyper_params["lr_scheduler"] == "step":
        return StepLR(optimizer, 
                      step_size=10,
                      gamma=0.5)
    elif hyper_params["lr_scheduler"] == "cyclic":
        return CyclicLR(optimizer, 
                        base_lr=hyper_params["lr"],
                        max_lr=0.003,
                        cycle_momentum=True if hyper_params["optimizer"] == "sgd" else False,
                        step_size_up=10)
    elif hyper_params["lr_scheduler"] == "cosineannealinglr":
        sched_1 = CosineAnnealingLR(optimizer, 
                                 T_max=hyper_params["epochs"] - hyper_params["lr_warmup_epochs"], 
                                 eta_min=hyper_params["lr_min"])
        
        if hyper_params["lr_warmup_epochs"] > 0:
            if hyper_params["lr_warmup_method"] == "linear":
                warmup_lr_scheduler = LinearLR(
                    optimizer, start_factor=hyper_params["lr_warmup_decay"], total_iters=hyper_params["lr_warmup_epochs"]
                )
            elif hyper_params["lr_warmup_method"] == "constant":
                warmup_lr_scheduler = ConstantLR(
                    optimizer, factor=hyper_params["lr_warmup_decay"], total_iters=hyper_params["lr_warmup_epochs"]
                )
            else:
                raise RuntimeError(
                    f"Invalid warmup lr method {hyper_params['lr_warmup_method']}. Only linear and constant are supported."
                )
            lr_scheduler = SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, sched_1], milestones=[hyper_params["lr_warmup_epochs"]]
            )
        else:
            lr_scheduler = sched_1
        
        return lr_scheduler
    elif hyper_params["lr_scheduler"] == "na":
        return None
    else:
        assert False, "Learning rate scheduler not implemented"


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups


class WarmUpLR():
    def __init__(self, optimizer, initial_lr, num_epochs_warm_up=10):
        """
        Args:
            optimizer: The used optimizer
            initial_lr: The initial learning rate
            num_epochs_warm_up (optional): The number of epochs the learning rate should be warmed up
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.num_epochs_warm_up = num_epochs_warm_up
        self.last_epoch = 0

    def step(self):
        if self.last_epoch < self.num_epochs_warm_up:
            lr = self.initial_lr / (self.num_epochs_warm_up - self.last_epoch)
        else:
            lr = self.initial_lr

        self.last_epoch += 1

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


# class ExponentialMovingAverage(AveragedModel):
#     """Maintains moving averages of model parameters using an exponential decay.
#     ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
#     `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
#     is used to compute the EMA.
#     """
#     def __init__(self, model, decay, device="cpu"):
#         def ema_avg(avg_model_param, model_param, num_averaged):
#             return decay * avg_model_param + (1 - decay) * model_param

#         super().__init__(model, device, ema_avg, use_buffers=True)


class Trainer:
    """Class for training a model.

     This class supports also logging with `TensorBoard`"""

    def __init__(self, val_set_loader, classes, args, num_epochs=None, train_set_loader=None,
                 experiment_name=None, hyper_params=None, start_epoch=0, num_epochs_early_stop=10, log_dir=None,
                 saved_models_dir=None, model_state_dict=None, optim_state_dict=None, scaler=None):
        """
        Args:
            model: Model to be trained
            criterion: Desired criterion
            optimizer: Desired optimizer
            scheduler: Learning rate scheduler. Set this argument to `None`,
            if you do not want to use an LR scheduler
            num_epochs: Maximum number of epochs the model should be trained for
            train_set_loader: `DataLoader` instance of the training set
            val_set_loader: `DataLoader` instance of the validation set
            experiment_name (optional): Name of the experiment (has to be a valid name for
            a directory). If set to `None`, the experiment will be named 'experiment_<unix time stamp>'
            hyper_params (optional): Dictionary containing the hyper parameters of the trained model to be
            logged to `TensorBoard`
            num_epochs_early_stop (optional): Number of epochs after the training should be stopped,
            if the validation loss does not improve any more
            log_dir (optional): Path to the root directory, where the `TensorBoard` data should be logged to.
            If set to `None`, no logging takes place.
            saved_models_dir (optional): Path to the root directory, where the models should be saved to.
            A model is saved after each epoch, where the validation loss improved compared to the best so far
            obtained validation loss. If set to `None`, no models are saved.
        """
        self.device = int(os.environ["LOCAL_RANK"])
        self.args = args
        self.hyper_params = hyper_params

        if args.amp:
            self.scaler = torch.cuda.amp.GradScaler() if args.amp else None
            if scaler:
                self.scaler = self.scaler.load_state_dict(scaler)

        # model
        self.model = create_model(self.args.model, 
                               num_classes=classes, 
                               weights=model_state_dict).to(self.device)
        
        # self.criterion = LabelSmoothingCrossEntropy().to(self.device)
        self.criterion = get_criterion(self.hyper_params).to(self.device)

        # decaying model weights
        parameters = set_weight_decay(
            self.model,
            args.decay,
            norm_weight_decay=args.norm_weight_decay,
            custom_keys_weight_decay=None,
        )

        self.optimizer = get_optimizer(parameters, 
                                       hyper_params=self.hyper_params)
        if optim_state_dict:
            self.optimizer.load_state_dict(optim_state_dict)
            
        # self.scheduler = WarmUpLR(optimizer=self.optimizer, initial_lr=self.hyper_params["lr"], num_epochs_warm_up=self.hyper_params["num_epochs_warmup"])
        self.scheduler = get_lr_scheduler(optimizer=self.optimizer, 
                                          hyper_params=self.hyper_params)

        # model with DDP
        self.model = DDP(self.model, device_ids=[self.device])

        # model_ema = None
        # if args.model_ema:
        #     # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        #     # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #     #
        #     # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        #     # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        #     # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        #     adjust = int(os.environ["WORLD_SIZE"]) * args.batch_sz * args.model_ema_steps / args.epochs
        #     alpha = 1.0 - args.model_ema_decay
        #     alpha = min(1.0, alpha * adjust)
        #     model_ema = ExponentialMovingAverage(self.model.module, 
        #                                          device=self.device, 
        #                                          decay=1.0 - alpha)

        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.train_set_loader = train_set_loader
        self.val_set_loader = val_set_loader
        self.num_epochs_early_stop = num_epochs_early_stop

        if experiment_name:
            self.experiment_name = experiment_name
        else:
            self.experiment_name = "experiment_" + str(int(time.time() * 1000.0))

        self.log_path = None
        self.summary_writer = None
        if log_dir:
            self.log_path = os.path.join(log_dir)

            os.makedirs(self.log_path, exist_ok=True)
            self.summary_writer = SummaryWriter(self.log_path)

        self.saved_models_path = None
        if saved_models_dir:
            self.saved_models_path = os.path.join(saved_models_dir)
            os.makedirs(self.saved_models_path, exist_ok=True)
    

    def _save_model(self, epoch, val_loss):
        if self.device == 0:
            logger.info(f"Saving model at {epoch}") 
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': val_loss,
                'scaler': self.scaler.state_dict(),
            }, os.path.join(self.saved_models_path, f"{self.experiment_name}.pth"))


    def __call__(self, *args, **kwargs):
        """Starts the training"""
        epoch_train_acc_1, epoch_train_acc_5, epoch_val_acc_1, epoch_val_acc_5, epoch_train_loss, epoch_val_loss = 0., 0., 0., 0., 0., 0.
        best_train_acc, best_val_acc, best_train_loss, best_val_loss = 0., 0., float('inf'), float('inf')

        early_stop_count = 0
        early_stop = False
        # epoch = 0

        for epoch in range(self.start_epoch, self.num_epochs):
            # logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            if self.train_set_loader:
                epoch_train_acc_1, epoch_train_acc_5, epoch_train_loss = self._train(epoch + 1)

            if self.scheduler:
                self.scheduler.step()

            if self.val_set_loader:
                epoch_val_acc_1, epoch_val_acc_5, epoch_val_loss = self._validate(epoch + 1)

            # logging
            if self.summary_writer:
                self.summary_writer.add_scalars("Accuracy-1", {
                    "training": epoch_train_acc_1,
                    "validation": epoch_val_acc_1,
                }, epoch + 1)
                self.summary_writer.add_scalars("Accuracy-5", {
                    "training": epoch_train_acc_5,
                    "validation": epoch_val_acc_5,
                }, epoch + 1)
                self.summary_writer.add_scalars("Loss", {
                    "training": epoch_train_loss,
                    "validation": epoch_val_loss,
                }, epoch + 1)
                self.summary_writer.add_scalar("Learning Rate", 
                                               self.optimizer.param_groups[0]['lr'], 
                                               epoch + 1)
                self.summary_writer.flush()

            if epoch_val_loss < best_val_loss:
                early_stop_count = 0
                self._save_model(epoch=epoch + 1, val_loss=best_val_loss)
            else:
                early_stop_count += 1

            best_train_acc_1 = (epoch_train_acc_1 if epoch_train_acc_1 > best_train_acc else best_train_acc)
            best_val_acc_1 = (epoch_val_acc_1 if epoch_val_acc_1 > best_val_acc else best_val_acc)
            best_train_acc_5 = (epoch_train_acc_5 if epoch_train_acc_5 > best_train_acc else best_train_acc)
            best_val_acc_5 = (epoch_val_acc_5 if epoch_val_acc_5 > best_val_acc else best_val_acc)
            best_train_loss = (epoch_train_loss if epoch_train_loss < best_train_loss else best_train_loss)
            best_val_loss = (epoch_val_loss if epoch_val_loss < best_val_loss else best_val_loss)

            if early_stop_count == self.num_epochs_early_stop:
                if self.device == 0:
                    logger.warning(f"Early stopping at epoch {epoch + 1} triggered.")
                early_stop = True
                break

        if self.summary_writer:
            if self.hyper_params and self.device == 0:
                self.summary_writer.add_hparams(
                    self.hyper_params,
                    {
                        "hparams/acc_1_train": best_train_acc_1,
                        "hparams/acc_1_val": best_val_acc_1,
                        "hparams/acc_5_train": best_train_acc_5,
                        "hparams/acc_5_val": best_val_acc_5,
                        "hparams/loss_train": best_train_loss,
                        "hparams/loss_val": best_val_loss,
                        "hparams/num_epochs": epoch + 1 if not early_stop else epoch + 1 - self.num_epochs_early_stop
                    }
                )
            self.summary_writer.close()
    

    def _train(self, epoch):
        # running_train_acc = 0
        running_train_acc_1 = 0
        running_train_acc_5 = 0
        running_train_loss = 0

        if torch.is_distributed:
            self.train_set_loader.sampler.set_epoch(epoch)

        self.model.train()
        train_bar = tqdm(self.train_set_loader, 
                         total=len(self.train_set_loader), 
                         colour="blue", 
                         file=sys.stdout, 
                         bar_format="{l_bar}{bar:30}{r_bar}",
                         disable=(self.device != 0))

        for data, label in train_bar:
            train_bar.set_description(f"Training [E= {epoch:02d} | L= {running_train_loss / data.size(0):.6f}]")
            data = data.to(device=self.device)
            label = label.to(device=self.device)

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                output = self.model(data)
                loss = self.criterion(output, label)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # running_train_acc += (output.argmax(dim=1) == label).float().mean()
            acc_1, acc_5 = accuracy(output, label, topk=(1, 5))
            running_train_acc_1 += acc_1.item()
            running_train_acc_5 += acc_5.item()
            running_train_loss += loss.item()

        # epoch_train_acc = running_train_acc / len(self.train_set_loader)
        epoch_train_acc_1 = running_train_acc_1 / len(self.train_set_loader)
        epoch_train_acc_5 = running_train_acc_5 / len(self.train_set_loader)
        epoch_train_loss = running_train_loss / len(self.train_set_loader)

        if self.device == 0:
            logger.info(f"[Train] E: {epoch:02d} | Acc-1: {epoch_train_acc_1:.4f} | Acc-5: {epoch_train_acc_5:.4f} | Loss: {epoch_train_loss:.6f}")

        return epoch_train_acc_1, epoch_train_acc_5, epoch_train_loss
    
    
    @torch.no_grad()
    def _validate(self, epoch=None):
        # running_val_acc = 0
        running_val_acc_1 = 0
        running_val_acc_5 = 0
        running_val_loss = 0

        self.model.eval()
        test_bar = tqdm(self.val_set_loader,  
                        total=len(self.val_set_loader), 
                        colour="green", 
                        file=sys.stdout, 
                        bar_format="{l_bar}{bar:30}{r_bar}",
                        disable=(self.device != 0))
        
        with torch.inference_mode():
            for data, label in test_bar:
                if epoch:
                    test_bar.set_description(f"Validation [E= {epoch:02d} | L= {running_val_loss / data.size(0):.6f}]")
                else:
                    test_bar.set_description(f"Validation [L= {running_val_loss / data.size(0):.6f}]")
                data = data.to(device=self.device)
                label = label.to(device=self.device)

                output = self.model(data)
                loss = self.criterion(output, label)

                acc_1, acc_5 = accuracy(output, label, topk=(1, 5))
                running_val_acc_1 += acc_1.item()
                running_val_acc_5 += acc_5.item()
                # running_val_acc += (output.argmax(dim=1) == label).float().mean()
                running_val_loss += loss.item()

        # epoch_val_acc = running_val_acc / len(self.val_set_loader)
        epoch_val_acc_1 = running_val_acc_1 / len(self.val_set_loader)
        epoch_val_acc_5 = running_val_acc_5 / len(self.val_set_loader)
        epoch_val_loss = running_val_loss / len(self.val_set_loader)

        if self.device == 0:
            if epoch:
                logger.info(f"[Valid] Epoch: {epoch:02d} | Acc-1: {epoch_val_acc_1:.4f} | Acc-5: {epoch_val_acc_5:.4f} | Loss: {epoch_val_loss:.6f}")
            else:
                logger.info(f"[TEST] | Acc-1: {epoch_val_acc_1:.4f} | Acc-5: {epoch_val_acc_5:.4f} | Loss: {epoch_val_loss:.6f}")

        return epoch_val_acc_1, epoch_val_acc_5, epoch_val_loss


    def test(self):
        epoch_val_acc_1, epoch_val_acc_5, epoch_val_loss = self._validate()
        return epoch_val_acc_1, epoch_val_acc_5, epoch_val_loss
