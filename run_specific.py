# multi_run.py (Grid Search 완성본)
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf
import datetime
import torch
import gc
import traceback
import itertools
import os

# [핵심] main.py에서 알맹이 함수 가져오기
from main import run_experiment
from run_settings import SGC_DICT


HP_SEARCH_SPACE = {

    # ======================

    # 1) CITESEER

    # ======================

    ("citeseer", "gcn"): {

        "layer": [2],                 # 2-layer 더 안정적

        "hidden": [128],

        "lr": [0.01],

        "dropout": [0.5],

        "wd": [5e-4, 5e-5],

    },

    ("citeseer", "gat"): {
        "layer": [2, 3],
        "hidden": [32, 64, 128, 192],
        "lr": [0.0005, 0.01, 0.001],
        "dropout": [0.3, 0.5, 0.2],
        "wd": [0, 5e-5, 5e-4, 0.001],
    },

    ("citeseer", "gin"): {

        "layer": [2],

        "hidden": [128],

        "lr": [0.001],

        "dropout": [0.2],

        "wd": [5e-5, 5e-4],

    },

    ("citeseer", "graphsage"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.001],

        "dropout": [0.5],

        "wd": [5e-5, 5e-4],

    },



    # ======================

    # 2) CORA

    # ======================

    ("cora", "gcn"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.01],

        "dropout": [0.2],

        "wd": [5e-5, 5e-4],

    },

    ("cora", "gat"): {
        "layer": [2, 3],
        "hidden": [32, 192, 64],
        "lr": [0.005, 0.001, 0.0005],
        "dropout": [0.2, 0.3, 0.5],
        "wd": [0, 5e-5, 5e-4],
    },

    ("cora", "gin"): {

        "layer": [2],

        "hidden": [128],

        "lr": [0.001],

        "dropout": [0.5],

        "wd": [5e-5, 5e-4],

    },

    ("cora", "graphsage"): {

        "layer": [3],

        "hidden": [64, 128],

        "lr": [0.01],

        "dropout": [0.5],

        "wd": [5e-4],

    },



    # ======================

    # 3) PUBMED

    # ======================

    ("pubmed", "gcn"): {

        "layer": [2],

        "hidden": [128],

        "lr": [0.01, 0.005],

        "dropout": [0.5],

        "wd": [5e-4],

    },

    ("pubmed", "gat"):  {
        "layer":   [2, 3],
        "hidden":  [64, 128],
        "lr":      [0.005, 0.01],
        "dropout": [0.2, 0.3, 0.5],
        "wd":      [5e-5, 5e-4],
    },

    ("pubmed", "gin"): {

        "layer": [2],

        "hidden": [64],

        "lr": [0.01],

        "dropout": [0.5],

        "wd": [5e-4, 5e-5],

    },

    ("pubmed", "graphsage"): {

        "layer": [2, 3],

        "hidden": [64],

        "lr": [0.001],

        "dropout": [0.2],

        "wd": [5e-4],

    },



    # ======================

    # 4) ACTOR

    # ======================

    ("actor", "gcn"): {

        "layer": [2],

        "hidden": [128],

        "lr": [0.001],

        "dropout": [0.5, 0.2],  # dropout 영향 명확히 긍정적

        "wd": [5e-4],

    },

    ("actor", "gat"): {
        "layer":   [2, 3],
        "hidden":  [64, 128, 192],
        "lr":      [0.005, 0.01, 0.002],
        "dropout": [0.2, 0.3, 0.5],
        "wd":      [0, 5e-5],
    },

    ("actor", "gin"): {

        "layer": [2],

        "hidden": [64],

        "lr": [0.01, 0.005],

        "dropout": [0.2],

        "wd": [5e-4],

    },

    ("actor", "graphsage"): {

        "layer": [3],   # actor에서 GraphSAGE는 3 layer 우세

        "hidden": [128],

        "lr": [0.005],

        "dropout": [0.5],

        "wd": [5e-5, 5e-4],

    },



    # ======================

    # 5) OGBN-PRODUCTS

    # ======================

    ("ogbn-products", "gcn"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.01],

        "dropout": [0.5],

        "wd": [5e-5],

    },

    ("ogbn-products", "gat"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.005],

        "dropout": [0.2],

        "wd": [5e-5],

    },

    ("ogbn-products", "gin"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.001],

        "dropout": [0.5],

        "wd": [5e-5],

    },

    ("ogbn-products", "graphsage"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.005],

        "dropout": [0.2],

        "wd": [5e-5],

    },



    # ======================

    # 6) OGBN-ARXIV

    # ======================

    ("ogbn-arxiv", "gcn"): {

        "layer": [3],

        "hidden": [256, 512],

        "lr": [0.01],

        "dropout": [0.3, 0.5],

        "wd": [5e-5, 0],

    },

    ("ogbn-arxiv", "gat"): {
        "layer":   [2, 3],
        "hidden":  [128, 64],
        "lr":      [0.005, 0.01],
        "dropout": [0.2, 0.5],
        "wd":      [0, 5e-5],
    },

    ("ogbn-arxiv", "gin"): {

        "layer": [2],

        "hidden": [256],

        "lr": [0.001],

        "dropout": [0.2],

        "wd": [5e-5, 0],

    },

    ("ogbn-arxiv", "graphsage"): {

        "layer": [3],

        "hidden": [256],

        "lr": [0.001, 0.005],

        "dropout": [0.2],

        "wd": [5e-5],

    },
    ("texas", "gcn"): {
        "layer": [2, 3],
        "hidden": [32, 64, 128, 256],
        "lr": [0.001, 0.005, 0.01, 0.02],
        "dropout": [0.3, 0.5, 0.7],
        "wd": [0, 1e-6, 5e-6, 1e-5, 5e-5, 5e-4]
    },

    ("texas", "graphsage"): {
        "layer": [2, 3],
        "hidden": [32, 64, 128, 256],
        "lr": [0.001, 0.005, 0.01, 0.02],
        "dropout": [0.3, 0.5, 0.7],
        "wd": [0, 1e-6, 5e-6, 1e-5, 5e-5, 5e-4]
    },

    ("cornell", "gcn"): {
        "layer": [2, 3],
        "hidden": [32, 64, 128, 256],
        "lr": [0.001, 0.005, 0.01, 0.02],
        "dropout": [0.3, 0.5, 0.7],
        "wd": [0, 1e-6, 5e-6, 1e-5, 5e-5, 5e-4]
    },

    ("cornell", "graphsage"): {
        "layer": [2, 3],
        "hidden": [32, 64, 128, 256],
        "lr": [0.001, 0.005, 0.01, 0.02],
        "dropout": [0.3, 0.5, 0.7],
        "wd": [0, 1e-6, 5e-6, 1e-5, 5e-5, 5e-4]
    },
    ("cornell", "sgc"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },
    ("cornell", "sgc"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },
    ("cornell", "sgc"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },
    ("cornell", "sgc"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },
    ("cornell", "sgc"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },
    ("cornell", "sgc"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },

}
HP_SEARCH_SPACE = SGC_DICT


# ============================================================
# 🔥 2) 공통: epoch 설정
# ============================================================
# EPOCH_TABLE = {
#     "ogbn-products": 4000,
#     "ogbn-arxiv": 2500,
#     "pubmed": 1000,
#     "actor": 1000,
#     "citeseer": 300,
#     "cora": 200,
#     "texas": 100,
#     "cornell": 100,
# }
# EPOCH_TABLE = {
#     "ogbn-products": 4000,
#     "ogbn-arxiv": 2500,
#     "pubmed": 150,
#     "actor": 100,
#     "citeseer": 100,
#     "cora": 100,
#     "texas": 50,
#     "cornell": 50,
# }
EPOCH_TABLE = {
    "ogbn-products": 2,
    "ogbn-arxiv": 5,
    "pubmed": 20,
    "actor": 50,
    "citeseer": 15,
    "cora": 15,
    "texas": 10,
    "cornell": 10,
}
PATIENCE_TABLE = {
    "ogbn-products": 300,
    "ogbn-arxiv": 200,
    "pubmed": 10,
    "actor": 150,
    "citeseer": 5,
    "cora": 5,
    "texas": 3,
    "cornell": 3,
}


# ============================================================
# 🔥 3) dataset/model 조합 단위로만 grid search 수행
# ============================================================
def run_grid_search(target_datasets=None, target_models=None):
    if not OmegaConf.has_resolver("kst"):
        OmegaConf.register_new_resolver(
            "kst",
            lambda fmt: datetime.datetime.now(
                datetime.timezone(datetime.timedelta(hours=9))
            ).strftime(fmt)
        )

    # 전체 대상 지정
    all_pairs = list(HP_SEARCH_SPACE.keys())

    # 만약 특정 dataset/model만 선택했다면 filtering
    if target_datasets is not None:
        all_pairs = [p for p in all_pairs if p[0] in target_datasets]
    if target_models is not None:
        all_pairs = [p for p in all_pairs if p[1] in target_models]

    print(f"🎯 Target pairs: {all_pairs}")

    with initialize(version_base=None, config_path="configs"):

        for (dataset, model) in all_pairs:
            # if dataset == "ogbn-products":
            #     continue

            hp = HP_SEARCH_SPACE[(dataset, model)]

            # Cartesian product for this dataset/model
            grid_keys = ["hidden", "lr", "dropout", "wd"]
            grid_values = [hp[k] for k in grid_keys]
            grid = list(itertools.product(*grid_values))

            print(f"\n===============================================")
            print(f"🚀 Running {dataset} / {model}")
            print(f"🔧 Total configs: {len(grid)}")
            print("===============================================\n")

            # epoch 설정
            epochs = EPOCH_TABLE[dataset]
            patience = PATIENCE_TABLE[dataset]

            # layer 개수대로 반복 (예: [2], [2,3])
            for layer in hp["layer"]:
                for i, combo in enumerate(grid):
                    hidden, lr, dropout, wd = combo

                    print(
                        f"[{dataset}/{model}] layer={layer} "
                        f"({i+1}/{len(grid)}) | "
                        f"hd={hidden}, lr={lr}, dr={dropout}, wd={wd}"
                    )

                    overrides = [
                        f"dataset={dataset}",
                        f"model={model}",
                        f"gpu_id=0",
                        "seed=5",

                        # HP 적용
                        f"model.num_layers={layer}",
                        f"model.hidden_dim={hidden}",
                        f"model.dropout={dropout}",
                        f"train.lr={lr}",
                        f"train.weight_decay={wd}",
                        f"train.patience={patience}",

                        # epochs
                        f"train.epochs={epochs}",
                    ]

                    # sampler 여부
                    is_large = dataset in ["ogbn-products"]
                    if is_large or (model == "graphsage" and dataset not in ["texas", "cornell"]):
                        overrides.append("dataset.use_sampler=true")
                        bs = 1024 if is_large else 512
                        overrides.append(f"sampler.batch_size={bs}")
                    else:
                        overrides.append("dataset.use_sampler=false")
                    if dataset == "ogbn-arxiv":
                        overrides.append("train.use_batchnorm=true")
                    else:
                        overrides.append("train.use_batchnorm=false")
                    if model == "sgc":
                        # overrides.append("train.use_early_stopping=false")
                        if dataset in ["texas", "cornell"]:
                            overrides.append("model.k_value=1")
                        elif dataset in ["cora", "citeseer"]:
                            overrides.append("model.k_value=2")
                        elif dataset == "pubmed":
                            overrides.append("model.k_value=3")


                    # layer-specific model sizes
                    if layer == 2:
                        overrides.append("model.sizes=[15,10]")
                    else:
                        overrides.append("model.sizes=[15,10,5]")

                    try:
                        if i == 0:
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                        cfg = compose(config_name="config", overrides=overrides)
                        run_experiment(cfg)

                    except Exception as e:
                        print(f"❌ Error in {dataset}/{model} @layer={layer}: {e}")
                        traceback.print_exc()

                    finally:
                        gc.collect()
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    gc.collect()
    torch.cuda.empty_cache()

    # 필요하면 일부만 선택 가능:
    # run_grid_search(target_datasets=["cora", "citeseer", "actor", "ogbn-arxiv"], target_models=["gat"])
    run_grid_search(target_models=["sgc"])

    # run_grid_search()
