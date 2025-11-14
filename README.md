
# GNN-Bench v2.1 (Config-Python + YAML Snapshot)

- Models: **GCN / GAT / GraphSAGE**
- Datasets: **Planetoid (Cora, Citeseer, PubMed)** + **OGB (ogbn-arxiv)**
- Logging: tqdm + pandas locally, wandb optionally
- python: 3.10
- **Python config** to define experiments (+ auto **YAML export** snapshot to `runs/.../config_dump_*.yaml`)

## Quickstart
```bash

pip install numpy==1.26.4 --force-reinstall
pip uninstall -y torch torchvision torchaudio

pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install pyg-lib torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
    
pip install -r requirements.txt

# run with a Python config (dynamic allowed)
python scripts/run_experiment.py configs/experiment_cora_gcn.py
python scripts/run_experiment.py configs/experiment_cora_gat.py
python scripts/run_experiment.py configs/experiment_arxiv_sage.py

numactl --physcpubind=1-6 python main.py


```
