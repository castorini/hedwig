from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class MPCNNEvaluatorFactory(object):
    """
    Get the corresponding Evaluator class for a particular dataset.
    """
    @staticmethod
    def get_evaluator(dataset_name, model, data_loader, batch_size, cuda):
        if data_loader is None:
            return None

        if dataset_name == 'sick':
            return SICKEvaluator(model, data_loader, batch_size, cuda)
        elif dataset_name == 'msrvid':
            return MSRVIDEvaluator(model, data_loader, batch_size, cuda)
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_name))


class Evaluator(object):
    """
    Evaluates performance of model on a Dataset, using metrics specific to the Dataset.
    """

    def __init__(self, model, data_loader, batch_size, cuda):
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.cuda = cuda

    def get_scores(self):
        """
        Get the scores used to evaluate the model.
        Should return ([score1, score2, ..], [score1_name, score2_name, ...]).
        The first score is the primary score used to determine if the model has improved.
        """
        raise NotImplementedError('Evaluator subclass needs to implement get_score')


class SICKEvaluator(Evaluator):

    def __init__(self, model, data_loader, batch_size, cuda):
        super(SICKEvaluator, self).__init__(model, data_loader, batch_size, cuda)

    def get_scores(self):
        self.model.eval()
        num_classes = self.data_loader.dataset.num_classes
        predict_classes = torch.arange(1, num_classes + 1).expand(self.batch_size, num_classes)
        if self.cuda:
            predict_classes = predict_classes.cuda()
        test_kl_div_loss = 0
        predictions = []
        true_labels = []
        for sentences, labels in self.data_loader:
            sent_a, sent_b = Variable(sentences['a'], volatile=True), Variable(sentences['b'], volatile=True)
            ext_feats = Variable(sentences['ext_feats'], volatile=True)
            labels = Variable(labels, volatile=True)
            output = self.model(sent_a, sent_b, ext_feats)
            test_kl_div_loss += F.kl_div(output, labels, size_average=False).data[0]
            # handle last batch which might have smaller size
            if len(predict_classes) != len(sent_a):
                predict_classes = torch.arange(1, num_classes + 1).expand(len(sent_a), num_classes)
                if self.cuda:
                    predict_classes = predict_classes.cuda()
            true_labels.append((predict_classes * labels.data).sum(dim=1))
            predictions.append((predict_classes * output.data.exp()).sum(dim=1))

            del output

        predictions = torch.cat(predictions).cpu().numpy()
        true_labels = torch.cat(true_labels).cpu().numpy()
        test_kl_div_loss /= len(self.data_loader.dataset)
        pearson_r = pearsonr(predictions, true_labels)[0]
        spearman_r = spearmanr(predictions, true_labels)[0]
        return [pearson_r, spearman_r, test_kl_div_loss], ['pearson_r', 'spearman_r', 'KL-divergence loss']


class MSRVIDEvaluator(Evaluator):

    def __init__(self, model, data_loader, batch_size, cuda):
        super(MSRVIDEvaluator, self).__init__(model, data_loader, batch_size, cuda)

    def get_scores(self):
        self.model.eval()
        num_classes = self.data_loader.dataset.num_classes
        predict_classes = torch.arange(0, num_classes).expand(self.batch_size, num_classes)
        if self.cuda:
            predict_classes = predict_classes.cuda()
        test_kl_div_loss = 0
        predictions = []
        true_labels = []
        for sentences, labels in self.data_loader:
            sent_a, sent_b = Variable(sentences['a'], volatile=True), Variable(sentences['b'], volatile=True)
            ext_feats = Variable(sentences['ext_feats'], volatile=True)
            labels = Variable(labels, volatile=True)
            output = self.model(sent_a, sent_b, ext_feats)
            test_kl_div_loss += F.kl_div(output, labels, size_average=False).data[0]
            # handle last batch which might have smaller size
            if len(predict_classes) != len(sent_a):
                predict_classes = torch.arange(0, num_classes).expand(len(sent_a), num_classes)
                if self.cuda:
                    predict_classes = predict_classes.cuda()
            true_labels.append((predict_classes * labels.data).sum(dim=1))
            predictions.append((predict_classes * output.data.exp()).sum(dim=1))

            del output

        predictions = torch.cat(predictions).cpu().numpy()
        true_labels = torch.cat(true_labels).cpu().numpy()
        test_kl_div_loss /= len(self.data_loader.dataset)
        pearson_r = pearsonr(predictions, true_labels)[0]
        return [pearson_r, test_kl_div_loss], ['pearson_r', 'KL-divergence loss']
