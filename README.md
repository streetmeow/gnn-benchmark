
# GNN-Bench v2.1 (Config-Python + YAML Snapshot)

- Models: **GCN / GAT / GraphSAGE**
- Datasets: **Planetoid (Cora, Citeseer, PubMed)** + **OGB (ogbn-arxiv)**
- Logging: tqdm + pandas locally, wandb optionally
- **Python config** to define experiments (+ auto **YAML export** snapshot to `runs/.../config_dump_*.yaml`)

## Quickstart
```bash
pip install -r requirements.txt

# run with a Python config (dynamic allowed)
python scripts/run_experiment.py configs/experiment_cora_gcn.py
python scripts/run_experiment.py configs/experiment_cora_gat.py
python scripts/run_experiment.py configs/experiment_arxiv_sage.py
```
