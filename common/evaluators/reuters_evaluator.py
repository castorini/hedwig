import torch
import torch.nn.functional as F
import numpy as np

from .evaluator import Evaluator


class ReutersEvaluator(Evaluator):

    def get_scores(self):
        self.model.eval()
        self.data_loader.init_epoch()
        n_dev_correct = 0
        total_loss = 0

        for batch_idx, batch in enumerate(self.data_loader):
            scores = self.model(batch.text[0], lengths=batch.text[1])
            scores_rounded = F.sigmoid(scores).round().long()

            # Using binary accuracy
            for tensor1, tensor2 in zip(scores_rounded, batch.label):
                if np.array_equal(tensor1, tensor2):
                    n_dev_correct += 1

            total_loss += F.binary_cross_entropy_with_logits(scores, batch.label.float(), size_average=False).item()

        accuracy = 100. * n_dev_correct / len(self.data_loader.dataset.examples)
        avg_loss = total_loss / len(self.data_loader.dataset.examples)

        return [accuracy, avg_loss], ['accuracy', 'cross_entropy_loss']
