import math
import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr

from .trainer import Trainer
from utils.serialization import save_checkpoint


class MSRVIDTrainer(Trainer):

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        # since MSRVID doesn't have validation set, we manually leave-out some training data for validation
        batches = math.ceil(len(self.train_loader.dataset.examples) / self.batch_size)
        start_val_batch = math.floor(0.8 * batches)
        left_out_val_a, left_out_val_b = [], []
        left_out_val_ext_feats = []
        left_out_val_labels = []

        for batch_idx, batch in enumerate(self.train_loader):
            # msrvid does not contain a validation set, we leave out some training data for validation to do model selection
            if batch_idx >= start_val_batch:
                left_out_val_a.append(batch.sentence_1)
                left_out_val_b.append(batch.sentence_2)
                left_out_val_ext_feats.append(batch.ext_feats)
                left_out_val_labels.append(batch.label)
                continue
            self.optimizer.zero_grad()

            # Select embedding
            sent1, sent2 = self.get_sentence_embeddings(batch)

            output = self.model(sent1, sent2, batch.ext_feats)
            loss = F.kl_div(output, batch.label, size_average=False)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, min(batch_idx * self.batch_size, len(batch.dataset.examples)),
                    len(batch.dataset.examples),
                    100. * batch_idx / (len(self.train_loader)), loss.item() / len(batch))
                )

        self.evaluate(self.train_evaluator, 'train')

        if self.use_tensorboard:
            self.writer.add_scalar('msrvid/train/kl_div_loss', total_loss / len(self.train_loader.dataset.examples), epoch)

        return left_out_val_a, left_out_val_b, left_out_val_ext_feats, left_out_val_labels

    def train(self, epochs):
        if self.lr_reduce_factor != 1 and self.lr_reduce_factor != None:
            scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.lr_reduce_factor, patience=self.patience)
        epoch_times = []
        prev_loss = -1
        best_dev_score = -1
        for epoch in range(1, epochs + 1):
            start = time.time()
            self.logger.info('Epoch {} started...'.format(epoch))
            left_out_a, left_out_b, left_out_ext_feats, left_out_label = self.train_epoch(epoch)

            # manually evaluating the validating set
            all_predictions, all_true_labels = [], []
            val_kl_div_loss = 0
            for i in range(len(left_out_a)):
                # Select embedding
                sent1 = self.embedding(left_out_a[i]).transpose(1, 2)
                sent2 = self.embedding(left_out_b[i]).transpose(1, 2)

                output = self.model(sent1, sent2, left_out_ext_feats[i])
                val_kl_div_loss += F.kl_div(output, left_out_label[i], size_average=False).item()
                predict_classes = left_out_a[i].new_tensor(torch.arange(0, self.train_loader.dataset.NUM_CLASSES))\
                                    .float().expand(len(left_out_a[i]), self.train_loader.dataset.NUM_CLASSES)

                predictions = (predict_classes * output.detach().exp()).sum(dim=1)
                true_labels = (predict_classes * left_out_label[i].detach()).sum(dim=1)
                all_predictions.append(predictions)
                all_true_labels.append(true_labels)

            predictions = torch.cat(all_predictions).cpu().numpy()
            true_labels = torch.cat(all_true_labels).cpu().numpy()
            pearson_r = pearsonr(predictions, true_labels)[0]
            val_kl_div_loss /= len(predictions)

            if self.use_tensorboard:
                self.writer.add_scalar('msrvid/dev/pearson_r', pearson_r, epoch)

            for param_group in self.optimizer.param_groups:
                self.logger.info('Validation size: %s Pearson\'s r: %s', output.size(0), pearson_r)
                self.logger.info('Learning rate: %s', param_group['lr'])

                if self.use_tensorboard:
                    self.writer.add_scalar('msrvid/lr', param_group['lr'], epoch)
                    self.writer.add_scalar('msrvid/dev/kl_div_loss', val_kl_div_loss, epoch)
                break

            if scheduler is not None:
                scheduler.step(pearson_r)

            end = time.time()
            duration = end - start
            self.logger.info('Epoch {} finished in {:.2f} minutes'.format(epoch, duration / 60))
            epoch_times.append(duration)

            if pearson_r > best_dev_score:
                best_dev_score = pearson_r
                save_checkpoint(epoch, self.model.arch, self.model.state_dict(), self.optimizer.state_dict(), best_dev_score, self.model_outfile)

            if abs(prev_loss - val_kl_div_loss) <= 0.0005:
                self.logger.info('Early stopping. Loss changed by less than 0.0005.')
                break

            prev_loss = val_kl_div_loss
            self.evaluate(self.test_evaluator, 'test')

        self.logger.info('Training took {:.2f} minutes overall...'.format(sum(epoch_times) / 60))