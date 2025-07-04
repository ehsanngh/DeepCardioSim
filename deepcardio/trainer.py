# Source: https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/training/trainer.py

import torch
from torch.cuda import amp
from torch import nn
from timeit import default_timer
from pathlib import Path
from typing import Union
import sys
import json
from .losses import LpLoss
from deepcardio.train_state import load_training_state, save_training_state


class Trainer:
    """
    A general Trainer class to train neural-operators on given datasets
    """
    def __init__(
        self,
        *,
        model: nn.Module,
        max_epochs: int,
        device: str='cpu',
        amp_autocast: bool=False,
        data_processor: nn.Module=None,
        eval_interval: int=1,
        log_output: bool=False,
        use_distributed: bool=False,
        verbose: bool=False,
    ):
        """
        Parameters
        ----------
        model : nn.Module
        max_epochs : int
        device : str 'cpu' or 'cuda'
        amp_autocast : bool, default is False
            whether to use torch.amp automatic mixed precision
        data_processor : DataProcessor class to transform data, default is None
            if not None, data from the loaders is transform first with data_processor.preprocess,
            then after getting an output from the model, that is transformed with data_processor.postprocess.
        eval_interval : int, default is 1
            how frequently to evaluate model and log training stats
        log_output : bool, default is False
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is False
        """

        self.epoch = 1
        self.best_loss = torch.tensor(1E+12, dtype=torch.float)
        self.max_epochs = max_epochs + 1
        self.eval_interval = eval_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.amp_autocast = amp_autocast
        
        if device == 'cpu':
            self.gpu_id = 'cpu'
        elif self.use_distributed:
            import os
            self.gpu_id = f'cuda:{int(os.environ["LOCAL_RANK"])}'
        else:
            self.gpu_id = 'cuda:0'

        self.model = model.to(self.gpu_id)
        self.data_processor = data_processor.to(self.gpu_id)

    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=None,
        training_loss=None,
        eval_losses=None,
        save_every: int=None,
        save_best: bool=False,
        best_loss_metric_key: str=None,
        save_dir: Union[str, Path]="./ckpt",
        resume_from_dir: Union[str, Path]=None,
    ):
        """Trains the given model on the given datasets.

        Parameters
        -----------
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        optimizer: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        save_every: int, optional, default is None
            if provided, interval at which to save checkpoints
        save_dir: str | Path, default "./ckpt"
            directory at which to save training states
        resume_from_dir: str | Path, default None
            if provided, resumes training state (model, 
            optimizer, regularizer, scheduler) from state saved in
            `resume_from_dir`
        
        Returns
        -------
        all_metrics: dict
            dictionary keyed f"{loader_name}_{loss_name}"
            of metric results for last validation epoch across
            all test_loaders
            
        """
        total_train_time_start = default_timer()
        self.list_epoch_metrics = []
        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        # attributes for checkpointing
        self.save_every = save_every
        if resume_from_dir is not None:
            self.resume_training_from_dir(resume_from_dir)
            sys.stdout.flush()

        if self.use_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[self.gpu_id])

        if self.verbose:
            print(f'[{self.gpu_id}] Training on {len(train_loader.dataset)} samples')
            print(f'[{self.gpu_id}] Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples '
                  f'on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()

        for epoch in range(self.epoch, self.max_epochs):
            train_err, avg_loss, avg_lasso_loss, epoch_train_time =\
                  self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(
                epoch=self.epoch,
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                epoch_train_time=epoch_train_time
            )
            
            if (epoch) % self.eval_interval == 0:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(epoch=epoch,
                                                 eval_losses=eval_losses,
                                                 test_loaders=test_loaders)
                if self.gpu_id in ['cuda:0', 'cpu']:
                    epoch_metrics.update(**eval_metrics)
                    self.list_epoch_metrics.append(epoch_metrics)
                    self.checkpoint(save_dir, save_best=False)
                    if save_best and \
                        (epoch_metrics[best_loss_metric_key] < self.best_loss):
                        self.best_loss = epoch_metrics[best_loss_metric_key]
                        self.checkpoint(save_dir, save_best=True)

            # Maximum training time is limited to 4 hours for each job submission
            if 13600 - (default_timer() - total_train_time_start) < 0:
                break
        
        print(f'[{self.gpu_id}] Total Training Time = {default_timer() - total_train_time_start}')
        return epoch_metrics

    def train_one_epoch(self, epoch, train_loader, training_loss):
        """train_one_epoch trains self.model on train_loader
        for one epoch and returns training metrics

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : torch.utils.data.DataLoader
            data loader of train examples
        test_loaders : dict
            dict of test torch.utils.data.DataLoader objects

        Returns
        -------
        all_errors
            dict of all eval metrics for the last epoch
        """
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0
        
        # track number of training examples in batch
        self.n_samples = 0

        b_sz = len(next(iter(train_loader)))
        
        if self.use_distributed:
            train_loader.sampler.set_epoch(epoch)

        for idx, sample in enumerate(train_loader):
            loss = self.train_one_batch(idx, sample, training_loss)
            loss.backward()
            self.optimizer.step()

            train_err += loss.item() * len(sample)
            with torch.no_grad():
                avg_loss += loss.item() * len(sample)
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss.item() * len(sample)

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        epoch_train_time = default_timer() - t1

        train_err /= len(train_loader.sampler)
        avg_loss /= self.n_samples
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None
        
        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
                batch_size=b_sz,
                len_trainDL=len(train_loader),
            )

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def evaluate_all(self, epoch, eval_losses, test_loaders):
        # evaluate and gather metrics across each loader in test_loaders
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            loader_metrics = self.evaluate(eval_losses, loader,
                                    log_prefix=loader_name, epoch=epoch)   
            all_metrics.update(**loader_metrics)
        self.log_eval(epoch=epoch,
                      eval_metrics=all_metrics)
        return all_metrics
    
    def evaluate(self, loss_dict, data_loader, log_prefix="", epoch=None):
        """Evaluates the model on a dictionary of losses

        Parameters
        ----------
        loss_dict : dict of functions
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary
        epoch : int | None
            current epoch. Used when logging both train and eval
            default None
        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """

        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()

        errors = {f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()}

        self.n_samples = 0
        if self.use_distributed:
            data_loader.sampler.set_epoch(epoch)
            
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                return_output = False
                if idx == len(data_loader) - 1:
                    return_output = True
                eval_step_losses, outs = self.eval_one_batch(
                    sample, loss_dict, return_output=return_output)

                for loss_name, val_loss in eval_step_losses.items():
                    errors[f"{log_prefix}_{loss_name}"] += val_loss.item() * len(sample)

        for key in errors.keys():
            errors[key] /= len(data_loader.sampler)  # self.n_samples
        
        return errors
    
    def on_epoch_start(self, epoch):
        """on_epoch_start runs at the beginning
        of each training epoch. This method is a stub
        that can be overwritten in more complex cases.

        Parameters
        ----------
        epoch : int
            index of epoch

        Returns
        -------
        None
        """
        self.epoch = epoch
        return None

    def train_one_batch(self, idx, sample, training_loss):
        """Run one batch of input through model
           and return training loss on outputs

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : dict
            data dictionary holding one batch

        Returns
        -------
        loss: float | Tensor
            float value of training loss
        """

        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()

        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.gpu_id)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        self.n_samples += sample["y"].shape[0]

        if self.amp_autocast:
            with amp.autocast(enabled=True):
                out = self.model(**sample)
        else:
            out = self.model(**sample)
        
        if self.epoch == 0 and idx == 0 and self.verbose:
            print(f"[{self.gpu_id}] Raw outputs of shape {out.shape}")

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        loss = 0.0

        if self.amp_autocast:
            with amp.autocast(enabled=True):
                loss += training_loss(out, **sample)
        else:
            loss += training_loss(out, **sample)

        if self.regularizer:
            loss += self.regularizer.loss
        
        return loss
    
    def eval_one_batch(self,
                       sample: dict,
                       eval_losses: dict,
                       return_output: bool=False):
        """eval_one_batch runs inference on one batch
        and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False
        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs
        """
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.gpu_id)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        self.n_samples += sample["y"].size(0)

        out = self.model(**sample)

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)
        
        eval_step_losses = {}

        for loss_name, loss in eval_losses.items():
            val_loss = loss(out, **sample)
            eval_step_losses[loss_name] = val_loss
        
        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None
    
    def log_training(self, 
            epoch:int,
            time: float,
            avg_loss: float,
            train_err: float,
            avg_lasso_loss: float=None,
            lr: float=None,
            batch_size: int=None,
            len_trainDL: int=None,
            ):
        """Basic method to log results
        from a single training epoch. 
        

        Parameters
        ----------
        epoch: int
        time: float
            training time of epoch
        avg_loss: float
            average train_err per individual sample
        train_err: float
            train error for entire epoch
        avg_lasso_loss: float
            average lasso loss from regularizer, optional
        lr: float
            learning rate at current epoch
        """

        msg = f"[{self.gpu_id}] Training: Epoch {epoch} time={time:.2f}, "
        msg += f"avg_loss={avg_loss:.4e}, "
        msg += f"train_err={train_err:.4e}, "
        msg += f"Batchsizes: {batch_size} | Steps: {len_trainDL}, "
        if avg_lasso_loss is not None:
            msg += f"avg_lasso={avg_lasso_loss:.4e}, "
        msg += f"lr={lr:.4e}"
        print(msg)
        sys.stdout.flush()
        
    
    def log_eval(self,
                 epoch: int,
                 eval_metrics: dict):
        """log_eval logs outputs from evaluation
        on all test loaders to stdout

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_metrics : dict
            metrics collected during evaluation
            keyed f"{test_loader_name}_{metric}" for each test_loader
       
        """
        msg = ""
        for metric, value in eval_metrics.items():
            if isinstance(value, float) or isinstance(value, torch.Tensor):
                msg += f"{metric}={value:.4e}, "   
        
        msg = f"[{self.gpu_id}] Eval: " + msg[:-2] # cut off last comma+space
        print(msg)
        sys.stdout.flush()


    def resume_training_from_dir(self, save_dir):
        """
        Resume training from save_dir created by `deepcardio.train_state`
        
        Params
        ------
        save_dir: Union[str, Path]
            directory in which training state is saved
            (see deepcardio.train_state)
        """
        
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        if isinstance(save_dir, Path):
            epoch, best_loss = load_training_state(
            save_dir=save_dir,
            model=self.model,
            load_best=False,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            regularizer=self.regularizer,
            map_location=self.gpu_id)

            self.epoch = epoch + 1
            self.best_loss = best_loss
            
            try:
                with open(save_dir.joinpath('metrics_dict.json').as_posix(), 'r') as f:
                    self.list_epoch_metrics = json.load(f)
                    print(f'[{self.gpu_id}] Resuming the training from epoch {self.epoch} ...')
            except FileNotFoundError:
                print(f"[{self.gpu_id}] The file {save_dir.joinpath('metrics_dict.json').as_posix()} does not exist."
                      "The previous epochs have not been loaded")
        return None

    def checkpoint(self, save_dir, save_best=False):
        """checkpoint saves current training state
        to a directory for resuming later.
        See deepcardio.training_state

        Parameters
        ----------
        save_dir : str | Path
            directory in which to save training state
        """
        
        save_training_state(
            save_dir=save_dir, 
            epoch=self.epoch,
            model=self.model,
            save_best=save_best,
            best_loss=self.best_loss,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            regularizer=self.regularizer
            )
        
        if save_best is False:
            save_dir = Path(save_dir)
            with open(save_dir.joinpath('metrics_dict.json').as_posix(), 'w') as f:
                json.dump(self.list_epoch_metrics, f, indent=4)

            if self.verbose:
                print(f"[{self.gpu_id}] Saved training metrics to {save_dir.joinpath('metrics_dict.json').as_posix()}")
        
        return None

       