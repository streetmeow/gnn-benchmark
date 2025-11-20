import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from experiment.train import BaseTrainer
from experiment.analyze import Evaluator


class KDTrainer(BaseTrainer):
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 evaluator: Evaluator,
                 device: torch.device,
                 # --- KD 전용 인자 ---
                 teacher_model: torch.nn.Module,
                 cfg_criterion: dict,  # Hydra config (e.g., kd_basic.yaml)
                 **kwargs):  # scheduler, logger 등을 받기 위함

        # BaseTrainer 초기화
        super().__init__(model, optimizer, evaluator, device, **kwargs)

        # Teacher 모델은 eval 모드로 고정
        self.teacher_model = teacher_model.to(self.device).eval()

        # Hydra config (kd_basic.yaml)에서 Loss 하이퍼파라미터 가져오기
        self.alpha = cfg_criterion.get("alpha", 0.1)
        self.T = cfg_criterion.get("T", 2.0)

    def _compute_loss(self, batch: Data) -> tuple[torch.Tensor, dict]:
        """
        [필수 구현]
        BaseTrainer의 '_compute_loss' 템플릿을 'Distillation' 전략으로 구현.
        """

        # 1. Student / Teacher Forward
        student_logits = self.model(batch.x, batch.edge_index)
        with torch.no_grad():
            teacher_logits = self.teacher_model(batch.x, batch.edge_index)

        # 2. Full-batch / Mini-batch 모드 자동 감지
        if hasattr(batch, 'batch_size'):
            s_target = student_logits[:batch.batch_size]
            t_target = teacher_logits[:batch.batch_size]
            labels = batch.y[:batch.batch_size]
        else:
            s_target = student_logits[batch.train_mask]
            t_target = teacher_logits[batch.train_mask]
            labels = batch.y[batch.train_mask]

        # 3. Loss 계산 (이 클래스의 핵심 '전략')

        # L_CE (vs Ground-truth)
        loss_ce = F.cross_entropy(s_target, labels)

        # L_KD (vs Teacher)
        loss_kd = (self.T * self.T) * F.kl_div(
            F.log_softmax(s_target / self.T, dim=-1),
            F.softmax(t_target / self.T, dim=-1),
            reduction='batchmean',
            log_target=False
        )

        # 4. Loss 조합
        total_loss = (self.alpha * loss_ce) + ((1 - self.alpha) * loss_kd)

        log_dict = {
            "loss_ce": loss_ce.item(),
            "loss_kd": loss_kd.item(),
            "loss_total": total_loss.item()
        }

        return total_loss, log_dict