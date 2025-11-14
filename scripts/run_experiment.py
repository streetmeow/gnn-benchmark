
import os, sys, importlib.util
from core.builder import build_all
from core.trainer import Trainer
from utils.config_io import dump_config_auto


def load_config_module(path: str):
    spec = importlib.util.spec_from_file_location("exp_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(config_path: str):
    mod = load_config_module(config_path)
    cfg = mod.get_config()
    os.makedirs(cfg.log_dir, exist_ok=True)
    yaml_path = dump_config_auto(cfg)
    print(f"[Config saved] {yaml_path}")
    objects = build_all(cfg)
    trainer = Trainer(**objects)
    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_experiment.py <path/to/config.py>")
        sys.exit(1)
    main(sys.argv[1])
