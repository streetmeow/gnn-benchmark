
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

pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 --index-url https://download.pytorch.org/whl/cu121 torchmetrics

pip install pyg-lib torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
    
pip install -r requirements.txt


numactl --physcpubind=1-6 python main.py
```


## Docker Execution
```bash
docker compose up -d
docker compose exec experiment /bin/bash
numactl --physcpubind=1-6 python main.py

# 종료 시
exit 

# 컨테이너 중지 및 재시작
docker compose stop
docker compose start

# 컨테이너 종료
docker compose down

# nvidia 인식 실패
docker compose down -v
docker compose up -d
docker compose exec experiment /bin/bash

# 업데이트
export DOCKER_TOKEN=your_dockerhub_token_here
echo "$DOCKER_TOKEN" | docker login -u streetmeow --password-stdin
docker push streetmeow/gnn-bench:v1.1
unset DOCKER_TOKEN
```

citeseer + gat, ogbn-arxiv + all model 에서 batch norm 이 우세.

다만 texas, cornell 은 batch norm 뿐만 아니라 sampling 하기에도 너무 작더라