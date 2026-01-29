import logging
from pathlib import Path
import time
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import AverageMeters, Metrics, ExcessMSELoss
from .linear_dataset import LinearSyntheticDataset

logger = logging.getLogger(__name__)

LossFn   = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MetricFn = Callable[[torch.Tensor, torch.Tensor], float]

METRICS_FILENAME = "metrics.json"
MODEL_FILENAME = "model.pth"


class BasicTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 train_criterion: LossFn,
                 test_loader: DataLoader,
                 test_criterion: LossFn,
                 optimizer: torch.optim.Optimizer | Callable[..., torch.optim.Optimizer],
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 device: torch.device = torch.device("cpu"),
                 seed: int = 0,
                 metric_fns: dict[str, MetricFn] = {},
                 grad_accumulation_steps: int = 1
                 ) -> None:

        self.model = model.to(device)
        self.train_loader = train_loader
        self.train_criterion = train_criterion.to(device)
        self.test_loader = test_loader
        self.test_criterion = test_criterion.to(device)

        self.device = device
        self.seed = seed
        self.metric_fns = metric_fns
        self.grad_accumulation_steps = grad_accumulation_steps

        logger.info(f"Using device: {device}")

        if callable(optimizer):
            # HACK: temp check, need to come up with a better check.
            if "Lora" in optimizer.func.__name__:
                optimizer = optimizer(self.model)
                logger.info(f"Using LoRA optimizer: {optimizer.__class__.__name__}")
            else:
                optimizer = optimizer(self.model.parameters())
        self.optimizer = optimizer

        if lr_scheduler is not None:
            lr_scheduler = lr_scheduler(self.optimizer)
        self.lr_scheduler = lr_scheduler

        # HACK: for linear synthetic dataset
        if isinstance(self.train_loader.dataset, LinearSyntheticDataset):
            min_loss = self.train_loader.dataset.fopt
            logger.info(f"Using min_loss = {min_loss} from linear dataset")
            loss_fn = ExcessMSELoss(reduction="sum", min_loss=min_loss).to(device)
            self.train_criterion = loss_fn
            self.test_criterion = loss_fn

        self.loss_reduction = "mean"
        if hasattr(self.train_criterion, "reduction"):
            self.loss_reduction = self.train_criterion.reduction
        # HACK: for OPLoRA optimizer
        if hasattr(self.optimizer, "loss_reduction"):
            self.optimizer.loss_reduction = self.loss_reduction

    def train(self) -> dict[str, float]:
        self.model.train()
        avg_meters = AverageMeters()
        should_zero_grad = True
        accum_batch_size = 0
        train_iterator = tqdm(self.train_loader, desc="Training", leave=False)

        for i, (inputs, labels) in enumerate(train_iterator):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            start_time = time.time()
            is_last_batch = (i + 1) == len(self.train_loader)
            should_accumulate = (i + 1) % self.grad_accumulation_steps != 0
            should_step = not should_accumulate or is_last_batch
            should_rescale_grad = self.loss_reduction == "mean" and self.grad_accumulation_steps > 1

            if should_zero_grad:
                self.optimizer.zero_grad()
                should_zero_grad = False
            outputs = self.model(inputs)
            loss = self.train_criterion(outputs, labels)
            if should_rescale_grad:
                (loss * inputs.size(0)).backward()  # scale up to sum for grad accumulation
            else:
                loss.backward()
            accum_batch_size += inputs.size(0)

            if should_step:
                if should_rescale_grad:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad /= accum_batch_size
                self.optimizer.step()
                should_zero_grad = True
                accum_batch_size = 0

                # HACK: hacky way to support SVDLoRA
                for m in self.model.modules():
                    if hasattr(m, "project_back"):
                        m.project_back()

            step_time = time.time() - start_time

            # Metrics
            avg_meters.update({"loss": loss.item()})
            for metric_key, metric_fn in self.metric_fns.items():
                metric_val = metric_fn(outputs, labels)
                avg_meters.update({metric_key: metric_val})
            avg_meters.update({"step_time": step_time})

        return avg_meters.asdict(prefix="train/")

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        avg_meters = AverageMeters()
        for inputs, labels in tqdm(self.test_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            start_time = time.time()
            outputs = self.model(inputs)
            loss = self.test_criterion(outputs, labels)
            step_time = time.time() - start_time

            avg_meters.update({"loss": loss.item()})
            for metric_key, metric_fn in self.metric_fns.items():
                metric_val = metric_fn(outputs, labels)
                avg_meters.update({metric_key: metric_val})
            avg_meters.update({"step_time": step_time})

        return avg_meters.asdict(prefix="test/")

    def scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def run(self,
            num_epochs: int,
            output_dir: Path,
            continue_training: bool = False,
            target_metric: str = "test_loss",
            save_model_every_epoch: bool = True
            ) -> float:
        # Load metrics and model
        metrics = Metrics()
        metrics_path = output_dir / METRICS_FILENAME
        model_path = output_dir / MODEL_FILENAME
        epochs_done = 0

        if continue_training:
            # Load recorded metrics and check last epoch
            if metrics_path.exists():
                metrics.load(metrics_path)
                epochs_done = metrics.get_last_index()
            if num_epochs <= epochs_done:
                logger.info("Run already done.")  # this helps skip run during sweeps
                return metrics.get(-1, target_metric)

            # Load model
            if model_path.exists():
                logger.info(f"Loading model from: {model_path}")
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
                logger.info(f"Continue training from epoch {epochs_done + 1}")
            else:
                logger.warning("continue_training was set to true but no model was found.")
                logger.info("Training from scratch.")
                metrics = Metrics()
                epochs_done = 0

        # Training loop
        for epoch in range(epochs_done + 1, num_epochs + 1):
            train_metrics = self.train()
            test_metrics = self.evaluate()
            self.scheduler_step()
            metrics.update(epoch, {**train_metrics, **test_metrics})
            metrics.save(metrics_path)
            logger.info(f"Train | Epoch {epoch}/{num_epochs} | " +\
                        '\t'.join([f"{key}: {val:.4f}" for key, val in train_metrics.items()]))
            logger.info(f"Test  | Epoch {epoch}/{num_epochs} | " +\
                        '\t'.join([f"{key}: {val:.4f}" for key, val in test_metrics.items()]))
            if save_model_every_epoch:
                torch.save(self.model.state_dict(), model_path)

        try:
            return metrics.get(-1, target_metric)
        except:
            return float('nan')


class TrainerWithClosure(BasicTrainer):
    def train(self) -> dict[str, float]:
        self.model.train()
        avg_meters = AverageMeters()
        for inputs, labels in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            def project_back(*args, **kwargs):
                for m in self.model.modules():
                    if hasattr(m, "project_back"):
                        m.project_back(*args, **kwargs)

            def closure():
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.train_criterion(outputs, labels)
                loss.backward()
                return loss

            start_time = time.time()
            # XXX: forwarding twice
            with torch.no_grad():
                outputs = self.model(inputs)
            loss = self.optimizer.step(closure)
            step_time = time.time() - start_time

            # Metrics
            avg_meters.update({"loss": loss.item()})
            for metric_key, metric_fn in self.metric_fns.items():
                metric_val = metric_fn(outputs, labels)
                avg_meters.update({metric_key: metric_val})
            avg_meters.update({"step_time": step_time})

        return avg_meters.asdict(prefix="train/")