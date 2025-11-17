# experiment/metrics.py

import torch
import torch.nn as nn
import torchmetrics


class Metrics(nn.Module):
    """
    GNN 벤치마크를 위한 동적 메트릭 계산기.
    config에 명시된 metric_names 리스트에 따라 필요한
    torchmetrics 객체만 'nn.ModuleDict'에 등록하여 관리.

    'inference_time'은 버퍼로 별도 관리.
    """

    def __init__(self, metric_names: list[str], num_classes: int):
        super().__init__()

        self.metric_names = metric_names
        self.metrics = nn.ModuleDict()

        # 1. Config에 명시된 torchmetrics 객체 동적 생성
        for name in metric_names:
            metric_obj = self._build_torchmetric(name, num_classes)
            if metric_obj:
                self.metrics[name] = metric_obj

        # 2. 'inference_time'은 버퍼로 별도 관리
        self.track_time = "inference_time" in metric_names
        if self.track_time:
            self.register_buffer("total_inference_time_s", torch.tensor(0.0))
            self.register_buffer("total_samples", torch.tensor(0.0))

    def _build_torchmetric(self, name: str, num_classes: int):
        """
        메트릭 이름(str)에 해당하는 torchmetrics 객체를 생성하는 팩토리.
        """
        if name == "acc":
            return torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
        elif name == "f1_macro":
            return torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            )
        elif name == "f1_micro":
            return torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="micro"
            )
        elif name == "precision":
            return torchmetrics.Precision(
                task="multiclass", num_classes=num_classes, average="macro"
            )
        elif name == "recall":
            return torchmetrics.Recall(
                task="multiclass", num_classes=num_classes, average="macro"
            )
        elif name == "average_loss":
            return torchmetrics.MeanMetric()
        elif name == "balanced_acc":
            return torchmetrics.BalancedAccuracy(
                task="multiclass", num_classes=num_classes
            )

        # 'inference_time'이나 알 수 없는 이름은 None 반환
        return None

    def update(self, logits: torch.Tensor, labels: torch.Tensor, batch_loss: float | None = None):
        """
        배치의 예측값과 레이블로 'nn.ModuleDict'의 모든 메트릭을 업데이트.

        Args:
            logits (torch.Tensor): 모델의 raw output (N, C)
            labels (torch.Tensor): 정답 레이블 (N,)
        """
        # 공통적으로 필요한 preds_classes는 한 번만 계산
        preds_classes = torch.argmax(logits, dim=-1)
        num_samples = labels.shape[0]

        # ModuleDict를 순회하며 각 메트릭 객체 업데이트
        # [수정 2] 'loss' 메트릭을 'batch_loss'로 업데이트하는 로직 추가
        for name, metric in self.metrics.items():
            if name == "loss":
                if batch_loss is not None:
                    # MeanMetric.update(value, weight)
                    # 배치 크기(num_samples)를 weight로 주어 가중 평균 계산
                    metric.update(batch_loss, weight=num_samples)
            else:
                # acc, f1 등
                metric.update(preds_classes, labels)

    def update_time(self, time_s: float, num_samples: int):
        """
        추론 시간 관련 버퍼를 업데이트.

        Args:
            time_s (float): 해당 배치 추론에 걸린 시간 (초)
            num_samples (int): 해당 배치의 샘플 수
        """
        if self.track_time:
            self.total_inference_time_s += time_s
            self.total_samples += num_samples

    def compute(self) -> dict:
        """
        누적된 상태를 바탕으로 최종 메트릭 딕셔너리를 계산.

        Returns:
            dict: {메트릭 이름: 계산된 값}
        """
        results = {}

        # 1. torchmetrics 결과 계산
        for name, metric in self.metrics.items():
            results[name] = metric.compute().item()

        # 2. 추론 시간 계산 (활성화된 경우)
        if self.track_time:
            if self.total_samples == 0:
                results["inference_time_avg_ms"] = 0.0
            else:
                avg_time_ms = (self.total_inference_time_s / self.total_samples).item() * 1000
                results["inference_time_avg_ms"] = avg_time_ms

        return results

    def reset(self):
        """
        모든 메트릭과 버퍼를 초기화.
        """
        # 1. torchmetrics 초기화
        for metric in self.metrics.values():
            metric.reset()

        # 2. 추론 시간 버퍼 초기화
        if self.track_time:
            self.total_inference_time_s.zero_()
            self.total_samples.zero_()