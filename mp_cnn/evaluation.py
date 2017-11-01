from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class MPCNNEvaluatorFactory(object):
    """
    Get the corresponding Evaluator class for a particular dataset.
    """
    @staticmethod
    def get_evaluator(dataset_cls, model, data_loader, batch_size, device):
        if data_loader is None:
            return None

        if hasattr(dataset_cls, 'NAME') and dataset_cls.NAME == 'sick':
            return SICKEvaluator(dataset_cls, model, data_loader, batch_size, device)
        elif hasattr(dataset_cls, 'NAME') and dataset_cls.NAME == 'msrvid':
            return MSRVIDEvaluator(dataset_cls, model, data_loader, batch_size, device)
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_cls))


class Evaluator(object):
    """
    Evaluates performance of model on a Dataset, using metrics specific to the Dataset.
    """

    def __init__(self, dataset_cls, model, data_loader, batch_size, device):
        self.dataset_cls = dataset_cls
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.device = device

    def get_scores(self):
        """
        Get the scores used to evaluate the model.
        Should return ([score1, score2, ..], [score1_name, score2_name, ...]).
        The first score is the primary score used to determine if the model has improved.
        """
        raise NotImplementedError('Evaluator subclass needs to implement get_score')


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
            output = self.model(batch.a, batch.b, batch.ext_feats)
            test_kl_div_loss += F.kl_div(output, batch.label, size_average=False).data[0]
            # handle last batch which might have smaller size
            if len(predict_classes) != len(batch.a):
                predict_classes = torch.arange(1, num_classes + 1).expand(len(batch.a), num_classes)

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


class MSRVIDEvaluator(Evaluator):

    def __init__(self, dataset_cls, model, data_loader, batch_size, device):
        super(MSRVIDEvaluator, self).__init__(dataset_cls, model, data_loader, batch_size, device)

    def get_scores(self):
        self.model.eval()
        num_classes = self.dataset_cls.NUM_CLASSES
        predict_classes = torch.arange(0, num_classes).expand(self.batch_size, num_classes)
        test_kl_div_loss = 0
        predictions = []
        true_labels = []

        for batch in self.data_loader:
            output = self.model(batch.a, batch.b, batch.ext_feats)
            test_kl_div_loss += F.kl_div(output, batch.label, size_average=False).data[0]
            # handle last batch which might have smaller size
            if len(predict_classes) != len(batch.a):
                predict_classes = torch.arange(0, num_classes).expand(len(batch.a), num_classes)

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

        return [pearson_r, test_kl_div_loss], ['pearson_r', 'KL-divergence loss']
