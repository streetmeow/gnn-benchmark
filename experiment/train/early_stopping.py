class EarlyStopping:
    def __init__(self, patience=100, mode='max', delta=1e-4):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.mode = mode
        self.delta = delta

    def step(self, metric: float, epoch: int) -> bool:
        score = metric if self.mode == 'max' else -metric

        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
