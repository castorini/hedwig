import math
import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr

from mp_cnn.trainers.trainer import Trainer


class MSRVIDTrainer(Trainer):

    def __init__(self, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):
        super(MSRVIDTrainer, self).__init__(model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)

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
            output = self.model(batch.sentence_1, batch.sentence_2, batch.ext_feats)
            loss = F.kl_div(output, batch.label)
            total_loss += loss.data[0]
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, min(batch_idx * self.batch_size, len(batch.dataset.examples)),
                    len(batch.dataset.examples),
                    100. * batch_idx / (len(self.train_loader)), loss.data[0])
                )

        self.evaluate(self.train_evaluator, 'train')

        if self.use_tensorboard:
            self.writer.add_scalar('msrvid/train/kl_div_loss', total_loss, epoch)

        return left_out_val_a, left_out_val_b, left_out_val_ext_feats, left_out_val_labels

    def train(self, epochs):
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
                output = self.model(left_out_a[i], left_out_b[i], left_out_ext_feats[i])
                val_kl_div_loss += F.kl_div(output, left_out_label[i], size_average=False).data[0]
                predict_classes = torch.arange(0, self.train_loader.dataset.NUM_CLASSES).expand(len(left_out_a[i]), self.train_loader.dataset.NUM_CLASSES)
                if self.train_loader.device != -1:
                    with torch.cuda.device(self.train_loader.device):
                        predict_classes = predict_classes.cuda()

                predictions = (predict_classes * output.data.exp()).sum(dim=1)
                true_labels = (predict_classes * left_out_label[i].data).sum(dim=1)
                all_predictions.append(predictions)
                all_true_labels.append(true_labels)

            predictions = torch.cat(all_predictions).cpu().numpy()
            true_labels = torch.cat(all_true_labels).cpu().numpy()
            pearson_r = pearsonr(predictions, true_labels)[0]
            val_kl_div_loss /= len(predictions)

            if self.use_tensorboard:
                self.writer.add_scalar('msrvid/dev/pearson_r', pearson_r, epoch)

            for param_group in self.optimizer.param_groups:
                self.logger.info('Validation size: %s Pearson\'s r: %s', output.size()[0], pearson_r)
                self.logger.info('Learning rate: %s', param_group['lr'])

                if self.use_tensorboard:
                    self.writer.add_scalar('msrvid/lr', param_group['lr'], epoch)
                    self.writer.add_scalar('msrvid/dev/kl_div_loss', val_kl_div_loss, epoch)
                break

            scheduler.step(pearson_r)

            end = time.time()
            duration = end - start
            self.logger.info('Epoch {} finished in {:.2f} minutes'.format(epoch, duration / 60))
            epoch_times.append(duration)

            if pearson_r > best_dev_score:
                best_dev_score = pearson_r
                torch.save(self.model, self.model_outfile)

            if abs(prev_loss - val_kl_div_loss) <= 0.0005:
                self.logger.info('Early stopping. Loss changed by less than 0.0005.')
                break

            prev_loss = val_kl_div_loss
            self.evaluate(self.test_evaluator, 'test')

        self.logger.info('Training took {:.2f} minutes overall...'.format(sum(epoch_times) / 60))
