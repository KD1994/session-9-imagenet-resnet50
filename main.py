import os
import logging
import logging.config
import argparse
import torch

from datetime import datetime
from src.dataset import get_data_loaders
from src.torch_utils import Trainer, ddp_setup, wrapup_ddp, get_hyperparams, list_optims, list_lr_schedulers, list_criterions
from src.utils import set_all_seeds, get_logging_schema, add_params_to_yaml


logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(prog=__file__,
                                     usage='%(prog)s [options]',
                                     description="Image Classification on Imagenet-1k using Resnet-50")
    # data handling
    parser.add_argument('--data-dir', type=str, default=os.path.join(os.getcwd(), "data"), required=True,
                        help='Path to where the data is stored.')
    parser.add_argument("--auto-augment", type=str, default=None, 
                        help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", type=int, default=9, 
                        help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", type=int, default=3, 
                        help="severity of augmix policy")
    parser.add_argument("--random-erase", type=float, default=0.0, 
                        help="random erasing probability (default: 0.0)")
    parser.add_argument("--val-resize-size", default=256, type=int, 
                        help="the resize size used for validation (default: 256)")
    parser.add_argument("--val-crop-size", default=224, type=int, 
                        help="the central crop size used for validation (default: 224)")
    parser.add_argument("--train-crop-size", default=224, type=int, 
                        help="the random crop size used for training (default: 224)")
    parser.add_argument("--ra-sampler", action="store_true", 
                        help="whether to use Repeated Augmentation in training")
    parser.add_argument("--ra-reps", default=3, type=int, 
                        help="number of repetitions for Repeated Augmentation (default: 3)")
    parser.add_argument('--batch-sz', type=int, default=16, required=False,
                        help='Batch size for train, validation & test dataset.')
    parser.add_argument('--data-workers', type=int, default=4, required=False,
                        help='number of workers to use in dataloader.')
    
    # Training parameters
    parser.add_argument("--amp", action="store_true", 
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--train', action="store_true", required=False,
                        help='Flag to initiate training.')
    parser.add_argument('--model', type=str, default="resnet50", 
                        help='Model architecture to train')
    parser.add_argument('--resume', action="store_true", required=False,
                        help='Flag to resume training from where stopped.')
    parser.add_argument("--criterion", choices=list_criterions(), default=list_criterions()[0],
                        help="Criterion for training.")
    parser.add_argument("--optim", choices=list_optims(), default=list_optims()[0],
                        help="Optimizer to use.")
    parser.add_argument("--lr-scheduler", choices=list_lr_schedulers(), default=list_lr_schedulers()[0],
                        help="LR scheduler to use.")
    parser.add_argument('--lr', type=float, default=5e-4, required=False,
                        help='Learning rate.')
    parser.add_argument("--lr-warmup-epochs", type=int, default=0, 
                        help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", type=str, default="constant", 
                        help="the warmup method (default: constant)")
    parser.add_argument("--lr-warmup-decay", type=float, default=0.01, 
                        help="the decay for lr")
    parser.add_argument("--lr-min", type=float, default=0.0, 
                        help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument('--decay', type=float, default=3e-2, required=False,
                        help='Weight decay.')
    parser.add_argument("--norm-weight-decay", type=float, default=None, 
                        help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument('--momentum', type=float, default=0.9, required=False,
                        help='LR momentum.')
    parser.add_argument("--warmup-epochs", type=int, default=10,
                        help="Number of epochs for learning rate warmup.")
    parser.add_argument("--patience-epochs", type=int, default=10,
                        help="Number of epochs after the training should be stopped, if the validation loss does not improve.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of maximum epochs to train.")
    
    # Model
    parser.add_argument('--weights', type=str, required=False,
                        help='Path to where model weights are saved.')
    # parser.add_argument("--model-ema", action="store_true", 
    #                     help="enable tracking Exponential Moving Average of model parameters")
    # parser.add_argument("--model-ema-steps", type=int, default=32,
    #                     help="the number of iterations that controls how often to update the EMA model (default: 32)")
    # parser.add_argument("--model-ema-decay", type=float, default=0.99998,
    #                     help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)")

    # experiment
    # parser.add_argument('--seed', type=int, default=42, required=True,
    #                     help='Seed for randomness.')
    parser.add_argument('--exp', type=str, default='exp-1', required=True,
                        help='experiment name')
    parser.add_argument('--debug', action="store_true", required=False,
                        help='Enable debugging mode.')
    
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # set_all_seeds(int(args.seed))
    logging.config.dictConfig(get_logging_schema(level="DEBUG" if args.debug else "INFO", 
                                                 file_name=os.path.join(os.getcwd(), 'runs', args.exp, f'logs', 'info.log')))
    logger.info(f"{args}")

    logger.debug(f"creating runs directory")
    os.makedirs(os.path.join(os.getcwd(), 'runs', args.exp), exist_ok=True)

    ddp_setup()

    logger.info(f"Preprocess data and store it..")
    train_loader, test_loader, classes = get_data_loaders(args)

    hy_params = get_hyperparams(args)
    
    if args.train:

        add_params_to_yaml(yml_path=os.path.join(os.getcwd(), 'runs', args.exp, 'logs', f"parameters-{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.yaml"), 
                       args=args)

        if args.resume:
            logger.info(f"Resume flag is on, loading model weights from the path {args.weights}")

            if not os.path.exists(os.path.join(args.weights)):
                logger.error(f"Model weights not found!")
                exit()
            
            ckpt = torch.load(args.weights)
            start_epoch = ckpt["epoch"]

            trainer = Trainer(start_epoch=start_epoch,
                              num_epochs=args.epochs,
                              train_set_loader=train_loader,
                              val_set_loader=test_loader,
                              experiment_name=args.exp,
                              hyper_params=hy_params,
                              model_state_dict=ckpt["model_state_dict"],
                              optim_state_dict=ckpt["optimizer_state_dict"],
                              scaler=ckpt["scaler"],
                              classes=classes,
                              num_epochs_early_stop=args.patience_epochs,
                              log_dir=os.path.join(os.getcwd(), 'runs', args.exp, 't-board'),
                              saved_models_dir=os.path.join(os.getcwd(), "runs", args.exp, "weights"),
                              args=args
                              )

        else:
            if args.weights:
                logger.info(f"initilizing model with weights from {args.weights}")

                if not os.path.exists(os.path.join(args.weights)):
                    logger.error(f"Model weights not found!")
                    exit()

                if args.weights.endswith(".pt") or args.weights.endswith(".pth"):
                    ckpt = torch.load(args.weights)
                    # start_epoch = ckpt["epoch"]
                    start_epoch = 0
                    model_state = ckpt["model_state_dict"]
                    scaler = ckpt["scaler"] if "scaler" in ckpt.keys() else None
                    # opt_state = ckpt["optimizer_state_dict"]
                    opt_state = None
                else:
                    assert False, "Specified weights are not acceptable!"

                trainer = Trainer(start_epoch=start_epoch,
                                  num_epochs=args.epochs,
                                  train_set_loader=train_loader,
                                  val_set_loader=test_loader,
                                  experiment_name=args.exp,
                                  hyper_params=hy_params,
                                  model_state_dict=model_state,
                                  optim_state_dict=opt_state,
                                  scaler=scaler,
                                  classes=classes,
                                  num_epochs_early_stop=args.patience_epochs,
                                  log_dir=os.path.join(os.getcwd(), 'runs', args.exp, 't-board'),
                                  saved_models_dir=os.path.join(os.getcwd(), "runs", args.exp, "weights"),
                                  args=args
                                  )

            else:
                logger.info(f"initilizing model with random weights")
                start_epoch = 0

                trainer = Trainer(start_epoch=start_epoch,
                                  num_epochs=args.epochs,
                                  train_set_loader=train_loader,
                                  val_set_loader=test_loader,
                                  experiment_name=args.exp,
                                  hyper_params=hy_params,
                                  classes=classes,
                                  num_epochs_early_stop=args.patience_epochs,
                                  log_dir=os.path.join(os.getcwd(), 'runs', args.exp, 't-board'),
                                  saved_models_dir=os.path.join(os.getcwd(), "runs", args.exp, "weights"),
                                  args=args
                                  )
        
        logger.info(f"Initiate training..")
        trainer()

        logger.info(f"Wrapping up DDP mode training")
        wrapup_ddp()

    else:
        logger.info(f"Testing mode enabled")

        logger.debug(f"creating test directory")
        os.makedirs(os.path.join(os.getcwd(), 'runs', args.exp, "test"), exist_ok=True)

        logger.info(f'Loading model weights from the {os.path.join(os.getcwd(), "runs", args.exp, "weights")}')

        if not os.path.exists(os.path.join(args.weights)):
            logger.error(f"Model weights not found!")
            exit()
        ckpt = torch.load(args.weights)
        
        trainer = Trainer(val_set_loader=test_loader,
                          experiment_name=args.exp,
                          hyper_params=hy_params,
                          model_state_dict=ckpt["model_state_dict"],
                          classes=classes,
                          num_epochs_early_stop=args.patience_epochs,
                          args=args
                          )
        logger.info(f"Initiate testing")
        with open(os.path.join(os.getcwd(), "runs", args.exp, "test", "stats.txt"), "a") as fp:
            acc_1, acc_5, loss = trainer.test()
            fp.write(f"Accuracy-1: {acc_1*100:.2f} | Accuracy-5: {acc_5*100:.2f} | Loss: {loss:.5f}\n")

    logger.info("Process completed.")


if __name__ == "__main__":
    main()
