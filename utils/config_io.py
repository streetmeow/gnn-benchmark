
import yaml, os
from dataclasses import asdict
from datetime import datetime

def dump_config(cfg, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(asdict(cfg), f, allow_unicode=True, sort_keys=False)

def dump_config_auto(cfg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    yaml_path = os.path.join(cfg.log_dir, f"config_dump_{timestamp}.yaml")
    dump_config(cfg, yaml_path)
    return yaml_path
