import numpy as np

from common.evaluators.evaluator import Evaluator
from utils.relevancy_metrics import get_map_mrr


class QAEvaluator(Evaluator):

    def __init__(self, dataset_cls, model, data_loader, batch_size, device):
        super(QAEvaluator, self).__init__(dataset_cls, model, None, data_loader, batch_size, device)

    def get_scores(self):
        self.model.eval()
        qids = []
        predictions = []
        labels = []

        for batch in self.data_loader:
            scores = self.model.convModel(batch.sentence_1, batch.sentence_2, batch.ext_feats)
            scores = self.model.linearLayer(scores)
            qid_array = np.transpose(batch.id.cpu().data.numpy())
            score_array = scores.cpu().data.numpy().reshape(-1)
            true_label_array = np.transpose(batch.label.cpu().data.numpy())

            qids.extend(qid_array.tolist())
            predictions.extend(score_array.tolist())
            labels.extend(true_label_array.tolist())

            del scores

        mean_average_precision, mean_reciprocal_rank = get_map_mrr(qids, predictions, labels, self.data_loader.device)

        return [mean_average_precision, mean_reciprocal_rank], ['map', 'mrr']
