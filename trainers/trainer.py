import os
import time
import logging
from pathlib import Path

import torch
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import tqdm
except ImportError:
    class _DummyTqdm:
        """Fallback no-op progress bar if tqdm isn't installed."""
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def __len__(self):
            return len(self.iterable)

        def set_postfix(self, *args, **kwargs):
            pass

    def tqdm(iterable, **kwargs):
        return _DummyTqdm(iterable, **kwargs)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        config,
    ):

        self.model = model.to(device)
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config = config

        self.epochs = config["epochs"]

        self.best_loss = 1e10

        self.use_amp = device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.writer = SummaryWriter(config["log_dir"])

        self.ckpt_dir = Path(config["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    ##########################################################

    def train(self):

        for epoch in range(self.epochs):

            t0 = time.time()

            train_loss = self.train_one_epoch(epoch)

            val_loss = self.validate(epoch)

            self.scheduler.step()

            dt = time.time() - t0

            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"lr={current_lr:.2e} | time={dt:.1f}s"
            )

            self.save_tensorboard_logs(epoch, train_loss, val_loss, current_lr)

            self.save_checkpoint(epoch, val_loss)

            # Save every N epochs
            if (epoch + 1) % self.config["save_every"] == 0:

                torch.save(
                    {
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                    },
                    self.ckpt_dir / f"epoch_{epoch+1}.pt",
                )

        self.writer.close()

    ##########################################################

    def train_one_epoch(self, epoch):

        self.model.train()

        running_loss = 0

        loader = tqdm(
            self.train_loader,
            desc=f"Train {epoch+1}",
        )

        for batch in loader:

            x = batch["input"].to(self.device)

            y = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            with autocast("cuda", enabled=self.use_amp):

                pred = self.model(x)

                loss_dict = self.criterion(
                    pred,
                    y,
                )

                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)

            self.scaler.update()

            running_loss += loss.item()

            loader.set_postfix(

                total=f"{loss.item():.4f}",

                l1=f"{loss_dict['l1'].item():.4f}",

                mse=f"{loss_dict['mse'].item():.4f}"

            )

        return running_loss / len(self.train_loader)

    ##########################################################

    @torch.no_grad()

    def validate(self, epoch):

        self.model.eval()

        running_loss = 0

        loader = tqdm(

            self.val_loader,

            desc=f"Val {epoch+1}",

        )

        for batch in loader:

            x = batch["input"].to(self.device)

            y = batch["target"].to(self.device)

            with autocast("cuda", enabled=self.use_amp):

                pred = self.model(x)

                loss_dict = self.criterion(
                    pred,
                    y,
                )

                loss = loss_dict["loss"]

            running_loss += loss.item()

            loader.set_postfix(

                total=f"{loss.item():.4f}",

                l1=f"{loss_dict['l1'].item():.4f}",

                mse=f"{loss_dict['mse'].item():.4f}"

            )

        return running_loss / len(self.val_loader)

    ##########################################################

    def save_checkpoint(
        self,
        epoch,
        val_loss,
    ):

        torch.save(

            {

                "epoch": epoch,

                "model": self.model.state_dict(),

                "optimizer": self.optimizer.state_dict(),

                "scheduler": self.scheduler.state_dict(),

                "best_loss": self.best_loss,

            },

            self.ckpt_dir / "last.pt",

        )

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            torch.save(

                {

                    "epoch": epoch,

                    "model": self.model.state_dict(),

                    "optimizer": self.optimizer.state_dict(),

                    "scheduler": self.scheduler.state_dict(),

                    "best_loss": self.best_loss,

                },

                self.ckpt_dir / "best.pt",

            )

            print()

            print("Best model saved.")

    ##########################################################

    def load_checkpoint(self, path):

        logger.info(f"Loading checkpoint from {path}")

        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model"])

        self.optimizer.load_state_dict(ckpt["optimizer"])

        self.scheduler.load_state_dict(ckpt["scheduler"])

        self.best_loss = ckpt.get("best_loss", 1e10)

        start_epoch = ckpt.get("epoch", 0) + 1

        logger.info(
            f"Resumed at epoch {start_epoch}, best_loss={self.best_loss:.4f}"
        )

        return start_epoch

    ##########################################################

    def save_tensorboard_logs(self, epoch, train_loss, val_loss, lr):

        self.writer.add_scalar("epoch/train_loss", train_loss, epoch)

        self.writer.add_scalar("epoch/val_loss", val_loss, epoch)

        self.writer.add_scalar("epoch/lr", lr, epoch)