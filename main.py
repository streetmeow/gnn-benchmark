import os
import hydra
from omegaconf import DictConfig
from data import GNNDataLoader


@hydra.main(version_base="1.2", config_path="configs", config_name="config")


def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    # Dataset build
    data, split_idx, num_classes, sampler = GNNDataLoader(cfg.dataset, cfg.sampler)

    # # Model build
    # student = build_model(
    #     cfg.student,
    #     num_features=data.x.size(1),
    #     num_classes=cfg.dataset.num_classes
    # )

    # Trainer build
    # trainer = Trainer(cfg.train, cfg.distill)

    # Run training
    # trainer.train(student, data, split_idx)


if __name__ == "__main__":
    main()
