
import os
import torch
from tqdm import tqdm
import pandas as pd
from utils.metrics import accuracy
from utils.wandb_logger import WandbRun

class Trainer:
    def __init__(self, model, optimizer, criterion, data_bundle, device, logger, cfg):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_bundle = data_bundle
        self.device = device
        self.logger = logger
        self.cfg = cfg
        self.data = data_bundle["data"].to(device)
        self.results = []
        os.makedirs(cfg.log_dir, exist_ok=True)
        self.wb = WandbRun(cfg)

    def fit(self):
        self.logger.info("Training started")
        for epoch in tqdm(range(1, self.cfg.train.epochs + 1), desc="Epochs", ncols=80):
            loss = self.train_one_epoch()
            train_acc = self.evaluate("train")
            val_acc = self.evaluate("val")
            test_acc = self.evaluate("test")
            row = dict(epoch=epoch, loss=loss, train_acc=train_acc, val_acc=val_acc, test_acc=test_acc)
            self.results.append(row)
            if self.wb.active:
                self.wb.log(row, step=epoch)
            if epoch % 10 == 0 or epoch == self.cfg.train.epochs:
                self.logger.info(f"[Epoch {epoch}] loss={loss:.4f} | val={val_acc:.3f} | test={test_acc:.3f}")
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.cfg.log_dir, "train_log.csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Training completed. Log saved to {csv_path}")
        self.wb.finish()

    def train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data)
        mask = self.data.train_mask
        loss = self.criterion(out[mask], self.data.y[mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, split="val"):
        self.model.eval()
        out = self.model(self.data)
        pred = out.argmax(dim=-1)
        mask = getattr(self.data, f"{split}_mask")
        return accuracy(self.data.y[mask], pred[mask])
