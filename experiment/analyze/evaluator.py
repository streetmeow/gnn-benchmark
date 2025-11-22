import torch
import torch.nn as nn
import time
from typing import Literal, Iterable

from experiment.analyze.metrics import Metrics


class Evaluator:
    """
    모델 평가(validation/test)를 수행하는 클래스.
    'full' (풀배치) 모드와 'mini' (미니배치) 모드를 모두 지원.

    - 'full' 모드: loader는 [data] 형태여야 하며, split_mask가 필수.
    - 'mini' 모드: loader는 NeighborLoader 등 PyG 미니배치 로더여야 함.
                   (batch 객체가 .batch_size 속성을 가진다고 가정)
    """

    def __init__(self, model: nn.Module, criterion: nn.Module, metrics: Metrics, device: torch.device):
        """
        Args:
            model (nn.Module): 평가할 GNN 모델 (BaseNN 상속)
            criterion (nn.Module): 손실 함수 (e.g., nn.CrossEntropyLoss)
            metrics (Metrics): Metrics(v3) 객체
            device (torch.device): 평가를 수행할 디바이스
        """
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.device = device

    def evaluate(self, loader: Iterable, mode: Literal["full", "mini"], split_mask: torch.Tensor | None = None,
                 return_logits: bool = False) -> tuple[dict, torch.Tensor | None]:
        """
        주어진 데이터로더를 사용해 모델을 평가하고 메트릭을 반환.

        Args:
            loader (Iterable):
                - mode='full': [data] (full Data 객체가 담긴 리스트)
                - mode='mini': NeighborLoader 등의 PyG 미니배치 로더
            mode (Literal["full", "mini"]): 평가 모드
            split_mask (torch.Tensor, optional):
                - mode='full'일 때 사용할 노드 마스크 (e.g., data.val_mask)
                - mode='mini'일 땐 무시됨.

        Returns:
            dict: self.metrics.compute() 결과 딕셔너리
        """
        self.model.eval()
        self.metrics.reset()
        logits_list = [] if return_logits else None

        # [Requirement 2]
        # 'loader'를 순회하는 구조 자체는 데이터가 풀배치든,
        # 미니배치든, 향후 추가될 서브그래프 배치든 동일하게 작동.
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                t_start = time.perf_counter()
                logits = self.model(batch.x, batch.edge_index)
                t_end = time.perf_counter()

                time_s = t_end - t_start

                # [Requirement 1] 풀배치/미니배치 분기 처리
                if mode == "full":
                    if split_mask is None:
                        raise ValueError("mode='full' requires 'split_mask'.")

                    mask = split_mask.to(self.device)

                    # 마스크를 적용하여 loss 및 메트릭 계산
                    target_labels = batch.y[mask].squeeze().long()
                    target_logits = logits[mask]
                    num_samples = mask.sum().item()

                elif mode == "mini":
                    if not hasattr(batch, 'batch_size'):
                        raise ValueError(
                            "mode='mini' requires batch to have 'batch_size' attribute (PyG NeighborLoader standard).")

                    # PyG NeighborLoader 표준: 타겟 노드는 0 ~ batch_size-1
                    target_logits = logits[:batch.batch_size]
                    target_labels = batch.y[:batch.batch_size].squeeze().long()

                    num_samples = batch.batch_size

                else:
                    raise ValueError(f"Unknown evaluation mode: {mode}")
                # 손실 계산
                loss = self.criterion(target_logits, target_labels)
                # 메트릭 업데이트
                self.metrics.update(logits=target_logits, labels=target_labels, batch_loss=loss.item())
                # 추론 시간 업데이트
                self.metrics.update_time(time_s=time_s, num_samples=num_samples)

                if return_logits:
                    logits_list.append(target_logits.cpu())
        metrics = self.metrics.compute()
        final_logits = None
        if return_logits and logits_list:
            final_logits = torch.cat(logits_list, dim=0)
        # 모든 배치 순회 후 최종 메트릭 계산
        return metrics, final_logits
