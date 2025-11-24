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
            metrics (Metrics): Metrics 객체
            device (torch.device): 평가를 수행할 디바이스
        """
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.device = device

    def evaluate(self, loader: Iterable, mode: Literal["full", "mini"], split_mask: torch.Tensor | None = None,
                    return_logits: bool = False) -> tuple[dict, torch.Tensor | None]:
        """
        데이터 로더를 사용해서 모델 평가를 수행
        1. inference time 측정
        2. 손실 및 메트릭 계산
        3. 필요시 로짓 반환
        """
        self.model.eval()
        self.metrics.reset()
        logits_list = [] if return_logits else None

        # loader 를 순회하며 평가 수행
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                # 추론 시간 측정
                t_start = time.perf_counter()
                logits = self.model(batch.x, batch.edge_index)
                t_end = time.perf_counter()

                time_s = t_end - t_start

                # 풀배치, 미니배치 모드에 따른 타겟 노드 선택
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
