from scripts import Logger


class BaseLogger(Logger):
    def extend_tags(self):
        self.tags.append("cpf_baseline_seed")
