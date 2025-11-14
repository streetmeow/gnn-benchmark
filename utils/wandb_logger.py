
import os
import wandb
from datetime import datetime
from utils.config_io import dump_config
import hydra

class WandbRun:
    """Thin wrapper to make wandb optional (auto-disabled if no API key or disabled in cfg)."""
    def __init__(self, cfg):
        self._active = False
        self._run = None
        if not cfg.wandb.enable:
            os.environ.setdefault("WANDB_MODE", "disabled")
            return

        api = os.environ.get("WANDB_API_KEY", "").strip()
        if not api and (cfg.wandb.mode is None):
            os.environ.setdefault("WANDB_MODE", "disabled")
        if cfg.wandb.mode:
            os.environ["WANDB_MODE"] = cfg.wandb.mode

        run_name = cfg.wandb.run_name or f"{cfg.data.name}_{cfg.model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            self._run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=run_name, config={}, tags=cfg.wandb.tags)
            self._active = wandb.run is not None
            if self._active:
                yaml_path = os.path.join(cfg.log_dir, "config_dump_wandb.yaml")
                dump_config(cfg, yaml_path)
                wandb.save(yaml_path)
        except Exception:
            os.environ["WANDB_MODE"] = "disabled"
            self._run = None
            self._active = False

    @property
    def active(self):
        return self._active

    def log(self, data: dict, step: int | None = None):
        if self._active:
            if step is not None:
                wandb.log(data, step=step)
            else:
                wandb.log(data)

    def finish(self):
        if self._active:
            wandb.finish()
            self._active = False
