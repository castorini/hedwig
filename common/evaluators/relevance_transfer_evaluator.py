import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm

from common.evaluators.evaluator import Evaluator
from datasets.bert_processors.robust45_processor import convert_examples_to_features
from utils.preprocessing import pad_input_matrix
from utils.tokenization import BertTokenizer

# Suppress warnings from sklearn.metrics
warnings.filterwarnings('ignore')


class RelevanceTransferEvaluator(Evaluator):

    def __init__(self, model, config, **kwargs):
        super().__init__(kwargs['dataset'], model, kwargs['embedding'], kwargs['data_loader'],
                         batch_size=config['batch_size'], device=config['device'])

        if config['model'] in {'BERT-Base', 'BERT-Large', 'HBERT-Base', 'HBERT-Large'}:
            variant = 'bert-large-uncased' if config['model'] == 'BERT-Large' else 'bert-base-uncased'
            self.tokenizer = BertTokenizer.from_pretrained(variant, is_lowercase=config['is_lowercase'])
            self.processor = kwargs['processor']
            if config['split'] == 'test':
                self.eval_examples = self.processor.get_test_examples(config['data_dir'], topic=config['topic'])
            else:
                self.eval_examples = self.processor.get_dev_examples(config['data_dir'], topic=config['topic'])

        self.config = config
        self.ignore_lengths = config['ignore_lengths']
        self.y_target = None
        self.y_pred = None
        self.docid = None

    def get_scores(self, silent=False):
        self.model.eval()
        self.y_target = list()
        self.y_pred = list()
        self.docid = list()
        total_loss = 0

        if self.config['model'] in {'BERT-Base', 'BERT-Large', 'HBERT-Base', 'HBERT-Large'}:
            eval_features = convert_examples_to_features(
                self.eval_examples,
                self.config['max_seq_length'],
                self.tokenizer,
                self.config['is_hierarchical']
            )

            unpadded_input_ids = [f.input_ids for f in eval_features]
            unpadded_input_mask = [f.input_mask for f in eval_features]
            unpadded_segment_ids = [f.segment_ids for f in eval_features]

            if self.config['is_hierarchical']:
                pad_input_matrix(unpadded_input_ids, self.config['max_doc_length'])
                pad_input_matrix(unpadded_input_mask, self.config['max_doc_length'])
                pad_input_matrix(unpadded_segment_ids, self.config['max_doc_length'])

            padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
            padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
            padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)
            label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            document_ids = torch.tensor([f.guid for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids, label_ids, document_ids)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.config['batch_size'])

            for input_ids, input_mask, segment_ids, label_ids, document_ids in tqdm(eval_dataloader, desc="Evaluating", disable=silent):
                input_ids = input_ids.to(self.config['device'])
                input_mask = input_mask.to(self.config['device'])
                segment_ids = segment_ids.to(self.config['device'])
                label_ids = label_ids.to(self.config['device'])

                with torch.no_grad():
                    logits = torch.sigmoid(self.model(input_ids, segment_ids, input_mask)).squeeze(dim=1)

                # Computing loss and storing predictions
                self.docid.extend(document_ids.cpu().detach().numpy())
                self.y_pred.extend(logits.cpu().detach().numpy())
                self.y_target.extend(label_ids.cpu().detach().numpy())
                loss = F.binary_cross_entropy(logits, label_ids.float())

                if self.config['n_gpu'] > 1:
                    loss = loss.mean()
                if self.config['gradient_accumulation_steps'] > 1:
                    loss = loss / self.config['gradient_accumulation_steps']
                total_loss += loss.item()

        else:
            self.data_loader.init_epoch()

            if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
                # Temporal averaging
                old_params = self.model.get_params()
                self.model.load_ema_params()

            for batch in tqdm(self.data_loader, desc="Evaluating", disable=silent):
                if hasattr(self.model, 'tar') and self.model.tar:
                    if self.ignore_lengths:
                        logits, rnn_outs = torch.sigmoid(self.model(batch.text)).squeeze(dim=1)
                    else:
                        logits, rnn_outs = torch.sigmoid(self.model(batch.text[0], lengths=batch.text[1])).squeeze(dim=1)
                else:
                    if self.ignore_lengths:
                        logits = torch.sigmoid(self.model(batch.text)).squeeze(dim=1)
                    else:
                        logits = torch.sigmoid(self.model(batch.text[0], lengths=batch.text[1])).squeeze(dim=1)

                total_loss += F.binary_cross_entropy(logits, batch.label.float()).item()
                if hasattr(self.model, 'tar') and self.model.tar:
                    # Temporal activation regularization
                    total_loss += (rnn_outs[1:] - rnn_outs[:-1]).pow(2).mean()

                self.docid.extend(batch.docid.cpu().detach().numpy())
                self.y_pred.extend(logits.cpu().detach().numpy())
                self.y_target.extend(batch.label.cpu().detach().numpy())

            if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
                # Temporal averaging
                self.model.load_params(old_params)

        predicted_labels = np.around(np.array(self.y_pred))
        target_labels = np.array(self.y_target)
        accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        average_precision = metrics.average_precision_score(target_labels, predicted_labels, average=None)
        f1 = metrics.f1_score(target_labels, predicted_labels, average='macro')
        avg_loss = total_loss / len(predicted_labels)

        try:
            precision = metrics.precision_score(target_labels, predicted_labels, average=None)[1]
        except IndexError:
            # Handle cases without positive labels
            precision = 0

        return [accuracy, precision, average_precision, f1, avg_loss], \
               ['accuracy', 'precision', 'average_precision', 'f1', 'cross_entropy_loss']
