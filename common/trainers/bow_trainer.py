import datetime
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from common.evaluators.bow_evaluator import BagOfWordsEvaluator
from datasets.bow_processors.abstract_processor import StreamingSparseDataset


class BagOfWordsTrainer(object):
    def __init__(self, model, vectorizer, optimizer, processor, args):
        self.args = args
        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        self.vectorizer = vectorizer

        train_examples = self.processor.get_train_examples(args.data_dir)
        self.train_features = vectorizer.fit_transform([x.text for x in train_examples])
        self.train_labels = [[float(x) for x in document.label] for document in train_examples]

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.snapshot_path = os.path.join(self.args.save_path, self.processor.NAME, '%s.pt' % timestamp)

        self.log_header = 'Epoch Iteration Progress   Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss'
        self.log_template = ' '.join('{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))

        self.train_loss = 0
        self.best_dev_f1 = 0
        self.nb_train_steps = 0
        self.unimproved_iters = 0
        self.early_stop = False

    def train_epoch(self, train_dataloader):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.model.train()
            self.optimizer.zero_grad()

            features, labels = tuple(t.to(self.args.device) for t in batch)
            logits = self.model(features)

            if self.args.n_gpu > 1:
                logits = logits.view(labels.size())

            if self.args.is_multilabel:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            else:
                loss = F.cross_entropy(logits, torch.argmax(labels, dim=1))

            if self.args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            self.optimizer.step()
            self.train_loss += loss.item()
            self.nb_train_steps += 1

    def train(self):
        train_data = StreamingSparseDataset(self.train_features, self.train_labels)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.args.batch_size)

        print("Number of examples: ", len(self.train_labels))
        print("Batch size:", self.args.batch_size)

        for epoch in trange(int(self.args.epochs), desc="Epoch"):
            self.train_epoch(train_dataloader)
            dev_evaluator = BagOfWordsEvaluator(self.model, self.vectorizer, self.processor, self.args, split='dev')
            dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = dev_evaluator.get_scores()[0]

            # Print validation results
            tqdm.write(self.log_header)
            tqdm.write(self.log_template.format(epoch + 1, self.nb_train_steps, epoch + 1, self.args.epochs,
                                                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss))

            # Update validation results
            if dev_f1 > self.best_dev_f1:
                self.unimproved_iters = 0
                self.best_dev_f1 = dev_f1
                torch.save(self.model, self.snapshot_path)
            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.args.patience:
                    self.early_stop = True
                    tqdm.write("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, self.best_dev_f1))
                    break
