import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from common.trainers.trainer import Trainer
from utils.nce_neighbors import get_nearest_neg_id, get_random_neg_id, get_batch

class QATrainer(Trainer):

    def __init__(self, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None, weighting=False):
        super(QATrainer, self).__init__(model, None, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
        self.loss = torch.nn.MarginRankingLoss(margin=1, size_average=True)
        self.question2answer = {}
        self.best_dev_map = 0
        self.best_dev_mrr = 0
        self.false_samples = {}
        self.question2answer = {}
        self.start = time.time()
        self.q2neg = {}
        self.iteration = 0
        self.name = self.train_loader.dataset.NAME
        self.dev_log_interval = trainer_config['dev_log_interval']
        self.neg_num = trainer_config['neg_num'] if 'neg_num' in trainer_config else 0
        self.neg_sample = trainer_config['neg_sample'] if 'neg_sample' in trainer_config else ''
        self.log_template = 'Train Epoch:{} [{}/{}]\tLoss:{}  Acc:{}'
        self.margin_label = trainer_config['margin_label']
        self.dev_index = 1
        self.weighting = weighting

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        acc = 0
        tot = 0
        for batch_idx, batch in enumerate(self.train_loader):
            self.iteration += 1

            features = self.model.convModel(batch.sentence_1, batch.sentence_2, batch.ext_feats)
            new_train_pos = {"answer": [], "question": [], "ext_feat": []}
            new_train_neg = {"answer": [], "question": [], "ext_feat": []}
            max_len_q = 0
            max_len_a = 0

            batch_near_list = []
            batch_qid = []
            batch_aid = []
            new_near_score = []

            for i in range(batch.batch_size):
                label_i = batch.label[i].cpu().data.numpy()[0]
                question_i = batch.sentence_1[i]
                # question_i = question_i[question_i!=1] # remove padding 1 <pad>
                answer_i = batch.sentence_2[i]
                # answer_i = answer_i[answer_i!=1] # remove padding 1 <pad>
                ext_feat_i = batch.ext_feats[i]
                qid_i = batch.id[i].data.cpu().numpy()[0]
                aid_i = batch.aid[i].data.cpu().numpy()[0]

                if qid_i not in self.question2answer:
                    self.question2answer[qid_i] = {"question": question_i, "pos": {}, "neg": {}}
                if label_i == 1:

                    if aid_i not in self.question2answer[qid_i]["pos"]:
                        self.question2answer[qid_i]["pos"][aid_i] = {}

                    self.question2answer[qid_i]["pos"][aid_i]["answer"] = answer_i
                    self.question2answer[qid_i]["pos"][aid_i]["ext_feat"] = ext_feat_i

                    # get neg samples in the first epoch but do not train
                    if epoch == 1:
                        continue
                    # random generate sample in the first training epoch
                    elif epoch == 2 or self.neg_sample == "random":
                        near_list = get_random_neg_id(self.q2neg, qid_i, k=self.neg_num)
                    else:
                        near_list, near_score = get_nearest_neg_id(features[i], self.question2answer[qid_i]["neg"],
                                                                   distance="cosine", k=self.neg_num, weight=True)
                        new_near_score.extend(near_score)

                    batch_near_list.extend(near_list)

                    neg_size = len(near_list)
                    if neg_size != 0:
                        answer_i = answer_i[answer_i != 1]  # remove padding 1 <pad>
                        question_i = question_i[question_i != 1]  # remove padding 1 <pad>
                        for near_id in near_list:
                            batch_qid.append(qid_i)
                            batch_aid.append(aid_i)

                            new_train_pos["answer"].append(answer_i)
                            new_train_pos["question"].append(question_i)
                            new_train_pos["ext_feat"].append(ext_feat_i)

                            near_answer = self.question2answer[qid_i]["neg"][near_id]["answer"]
                            if question_i.size(0) > max_len_q:
                                max_len_q = question_i.size(0)
                            if near_answer.size(0) > max_len_a:
                                max_len_a = near_answer.size(0)
                            if answer_i.size(0) > max_len_a:
                                max_len_a = answer_i.size(0)

                            ext_feat_neg = self.question2answer[qid_i]["neg"][near_id]["ext_feat"]
                            new_train_neg["answer"].append(near_answer)
                            new_train_neg["question"].append(question_i)
                            new_train_neg["ext_feat"].append(ext_feat_neg)

                elif label_i == 0:

                    if aid_i not in self.question2answer[qid_i]["neg"]:
                        answer_i = answer_i[answer_i != 1]
                        self.question2answer[qid_i]["neg"][aid_i] = {"answer": answer_i}

                    if "ext_feat" in self.question2answer[qid_i]["neg"][aid_i]:
                        del self.question2answer[qid_i]["neg"][aid_i]["ext_feat"]
                    self.question2answer[qid_i]["neg"][aid_i]["feature"] = features[i].data.cpu().numpy()
                    self.question2answer[qid_i]["neg"][aid_i]["ext_feat"] = ext_feat_i


                    if epoch == 1:
                        if qid_i not in self.q2neg:
                            self.q2neg[qid_i] = []

                        self.q2neg[qid_i].append(aid_i)

            del features
            # pack the selected pos and neg samples into the torchtext batch and train
            if epoch != 1:
                true_batch_size = len(new_train_neg["answer"])
                if true_batch_size != 0:
                    for j in range(true_batch_size):
                        new_train_neg["answer"][j] = F.pad(new_train_neg["answer"][j],
                                                           (0, max_len_a - new_train_neg["answer"][j].size(0)),
                                                           value=1)
                        new_train_pos["answer"][j] = F.pad(new_train_pos["answer"][j],
                                                           (0, max_len_a - new_train_pos["answer"][j].size(0)),
                                                           value=1)
                        new_train_pos["question"][j] = F.pad(new_train_pos["question"][j],
                                                             (0, max_len_q - new_train_pos["question"][j].size(0)),
                                                             value=1)
                        new_train_neg["question"][j] = F.pad(new_train_neg["question"][j],
                                                             (0, max_len_q - new_train_neg["question"][j].size(0)),
                                                             value=1)


                    pos_batch = get_batch(new_train_pos["question"], new_train_pos["answer"], new_train_pos["ext_feat"],
                                          true_batch_size)
                    neg_batch = get_batch(new_train_neg["question"], new_train_neg["answer"], new_train_neg["ext_feat"],
                                          true_batch_size)

                    self.model.train()
                    self.optimizer.zero_grad()
                    output = self.model([pos_batch, neg_batch])

                    cmp = output[:, 0] > output[:, 1]
                    acc += sum(cmp.data.cpu().numpy())
                    tot += true_batch_size

                    loss = self.loss(output[:, 0], output[:, 1], self.margin_label)
                    if len(new_near_score) != 0 and self.weighting:
                        # Element wise weighting hasn't been implemented for MarginRankingLoss by pytorch
                        # Apply the mean score and set branch size to 1 as weighting implementation temporarily here
                        # Reference: https://github.com/pytorch/pytorch/issues/264
                        # loss *= torch.autograd.variable.Variable(torch.from_numpy(np.array(new_near_score)).cuda())
                        loss *= np.mean(new_near_score)

                    loss_num = loss.data.cpu().numpy()[0]
                    total_loss += loss_num
                    loss.backward()
                    self.optimizer.step()

                    del new_train_neg
                    del new_train_pos

                    if self.iteration % self.dev_log_interval == 1 and epoch != 1:
                        dev_map, dev_mrr = self.evaluate(self.dev_evaluator, 'dev')
                        test_map, test_mrr = self.evaluate(self.test_evaluator, 'test')

                        if self.use_tensorboard:
                            self.writer.add_scalar('{}/dev/map'.format(self.name), dev_map, self.dev_index)
                            self.writer.add_scalar('{}/dev/mrr'.format(self.name), dev_mrr, self.dev_index)
                            self.writer.add_scalar('{}/test/map'.format(self.name), test_map, self.dev_index)
                            self.writer.add_scalar('{}/test/mrr'.format(self.name), test_mrr, self.dev_index)
                            self.writer.add_scalar('{}/train/loss'.format(self.name), loss_num, self.dev_index)
                            self.writer.add_scalar('{}/lr'.format(self.train_loader.dataset.NAME),
                                                   self.optimizer.param_groups[0]['lr'], self.dev_index)

                        self.dev_index += 1
                        if self.best_dev_mrr < dev_mrr:
                            torch.save(self.model, self.model_outfile)
                            self.best_dev_mrr = dev_mrr
                            self.best_dev_map = dev_map

                    if self.iteration % self.log_interval == 1 and epoch != 1:
                        # logger.info progress message
                        self.logger.info(self.log_template.format(epoch, min(batch_idx * self.batch_size, len(batch.dataset.examples)),
                                                                  len(batch.dataset.examples), loss_num, acc / tot))

        return total_loss

    def train(self, epochs):

        scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.lr_reduce_factor, patience=self.patience)
        epoch_times = []
        self.start = time.time()
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss = self.train_epoch(epoch)

            end = time.time()
            duration = end - start
            self.logger.info('Epoch {} finished in {:.2f} minutes'.format(epoch, duration / 60))
            epoch_times.append(duration)

            scheduler.step(train_loss)

        self.logger.info('Training took {:.2f} minutes overall...'.format(sum(epoch_times) / 60))
