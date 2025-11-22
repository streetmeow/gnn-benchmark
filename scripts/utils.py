import torch


class FeatureExtractor:
    """
    모델의 특정 레이어(layer_names) 출력을 낚아채는 Context Manager
    """
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.hooks = []

    def __enter__(self):
        # 모델의 모든 모듈을 순회하며 이름이 일치하면 Hook 설치
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                # forward가 끝날 때(output) 값을 features에 저장
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: self.features.update({n: o.detach().cpu()})
                )
                self.hooks.append(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 사용이 끝나면 Hook 제거 (메모리 누수 방지)
        for hook in self.hooks:
            hook.remove()