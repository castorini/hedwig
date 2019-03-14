import torch
import torch.nn.functional as F

from .evaluator import Evaluator


class SNLIEvaluator(Evaluator):

    def get_scores(self):
        self.model.eval()
        test_kl_div_loss = 0
        acc_total = 0

        for batch in self.data_loader:
            # Select embedding
            sent1, sent2 = self.get_sentence_embeddings(batch)

            output = self.model(sent1, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw)
            test_kl_div_loss += F.kl_div(output, batch.label, size_average=False).item()

            true_label = torch.max(batch.label.data, 1)[1]
            prediction = torch.max(output, 1)[1]
            acc_total += ((true_label == prediction)).sum().item()

            del output

        test_kl_div_loss /= len(batch.dataset.examples)

        accuracy = acc_total / len(self.data_loader.dataset.examples)

        return [accuracy, test_kl_div_loss], ['accuracy', 'KL-divergence loss']

