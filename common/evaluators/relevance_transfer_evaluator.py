import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from common.evaluators.evaluator import Evaluator

# Suppress warnings from sklearn.metrics
warnings.filterwarnings('ignore')


class RelevanceTransferEvaluator(Evaluator):

    def __init__(self, dataset_cls, model, embedding, data_loader, batch_size, device, keep_results=False):
        super().__init__(dataset_cls, model, embedding, data_loader, batch_size, device, keep_results)
        self.ignore_lengths = False
        self.y_target = None
        self.y_pred = None
        self.docid = None

    def get_scores(self):
        self.model.eval()
        self.data_loader.init_epoch()
        self.y_target = list()
        self.y_pred = list()
        self.docid = list()
        total_loss = 0

        if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
            # Temporal averaging
            old_params = self.model.get_params()
            self.model.load_ema_params()

        for batch_idx, batch in enumerate(self.data_loader):
            if hasattr(self.model, 'tar') and self.model.tar:
                if self.ignore_lengths:
                    scores, rnn_outs = self.model(batch.text)
                else:
                    scores, rnn_outs = self.model(batch.text[0], lengths=batch.text[1])
            else:
                if self.ignore_lengths:
                    scores = self.model(batch.text)
                else:
                    scores = self.model(batch.text[0], lengths=batch.text[1])

            # Computing loss and storing predictions
            predictions = torch.sigmoid(scores).squeeze(dim=1)
            total_loss += F.binary_cross_entropy(predictions, batch.label.float()).item()
            self.docid.extend(batch.docid.cpu().detach().numpy())
            self.y_pred.extend(predictions.cpu().detach().numpy())
            self.y_target.extend(batch.label.cpu().detach().numpy())

            if hasattr(self.model, 'tar') and self.model.tar:
                # Temporal activation regularization
                total_loss += (rnn_outs[1:] - rnn_outs[:-1]).pow(2).mean()

        predicted_labels = np.around(np.array(self.y_pred))
        target_labels = np.array(self.y_target)
        accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        average_precision = metrics.average_precision_score(target_labels, predicted_labels, average=None)
        f1 = metrics.f1_score(target_labels, predicted_labels, average='macro')
        avg_loss = total_loss / len(self.data_loader.dataset.examples)

        try:
            precision = metrics.precision_score(target_labels, predicted_labels, average=None)[1]
        except IndexError:
            # Handle cases without positive labels
            precision = 0

        if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
            # Temporal averaging
            self.model.load_params(old_params)

        return [accuracy, precision, average_precision, f1, avg_loss], ['accuracy', 'precision', 'average_precision', 'f1', 'cross_entropy_loss']