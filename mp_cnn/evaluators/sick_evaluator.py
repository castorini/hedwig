from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F

from mp_cnn.evaluators.evaluator import Evaluator


class SICKEvaluator(Evaluator):

    def __init__(self, dataset_cls, model, data_loader, batch_size, device):
        super(SICKEvaluator, self).__init__(dataset_cls, model, data_loader, batch_size, device)

    def get_scores(self):
        self.model.eval()
        num_classes = self.dataset_cls.NUM_CLASSES
        predict_classes = torch.arange(1, num_classes + 1).expand(self.batch_size, num_classes)
        test_kl_div_loss = 0
        predictions = []
        true_labels = []

        for batch in self.data_loader:
            output = self.model(batch.sentence_1, batch.sentence_2, batch.ext_feats)
            test_kl_div_loss += F.kl_div(output, batch.label, size_average=False).data[0]
            # handle last batch which might have smaller size
            if len(predict_classes) != len(batch.sentence_1):
                predict_classes = torch.arange(1, num_classes + 1).expand(len(batch.sentence_1), num_classes)

            if self.data_loader.device != -1:
                with torch.cuda.device(self.device):
                    predict_classes = predict_classes.cuda()

            true_labels.append((predict_classes * batch.label.data).sum(dim=1))
            predictions.append((predict_classes * output.data.exp()).sum(dim=1))

            del output

        predictions = torch.cat(predictions).cpu().numpy()
        true_labels = torch.cat(true_labels).cpu().numpy()
        test_kl_div_loss /= len(batch.dataset.examples)
        pearson_r = pearsonr(predictions, true_labels)[0]
        spearman_r = spearmanr(predictions, true_labels)[0]

        return [pearson_r, spearman_r, test_kl_div_loss], ['pearson_r', 'spearman_r', 'KL-divergence loss']
