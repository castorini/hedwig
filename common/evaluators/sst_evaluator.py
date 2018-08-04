import torch
import torch.nn.functional as F

from .evaluator import Evaluator


class SSTEvaluator(Evaluator):

    def get_scores(self):
        self.model.eval()
        self.data_loader.init_epoch()
        n_dev_correct = 0
        total_loss = 0

        for batch_idx, batch in enumerate(self.data_loader):
            scores = self.model(batch.text)
            n_dev_correct += (
                torch.max(scores, 1)[1].view(batch.label.size()).data == batch.label.data).sum().item()
            total_loss += F.cross_entropy(scores, batch.label, size_average=False).item()

        accuracy = 100. * n_dev_correct / len(self.data_loader.dataset.examples)
        avg_loss = total_loss / len(self.data_loader.dataset.examples)

        return [accuracy, avg_loss], ['accuracy', 'cross_entropy_loss']
