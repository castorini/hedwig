from collections import namedtuple

from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils

import data
import model as mod

Context = namedtuple("Context", "model, train_loader, dev_loader, test_loader, optimizer, criterion, params")
EvaluateResult = namedtuple("EvaluateResult", "pearsonr, spearmanr")

def create_context(config):
    def collate_fn(batch):
        emb1 = []
        emb2 = []
        labels = []
        for s1, s2, l in batch:
            emb1.append(s1)
            emb2.append(s2)
            labels.append(l)
        emb1 = torch.LongTensor(emb1)
        emb2 = torch.LongTensor(emb2)
        labels = torch.Tensor(labels)
        emb1 = torch.autograd.Variable(emb1, requires_grad=False)
        emb2 = torch.autograd.Variable(emb2, requires_grad=False)
        labels = torch.autograd.Variable(labels, requires_grad=False)
        if not config.cpu:
            emb1 = emb1.cuda()
            emb2 = emb2.cuda()
            labels = labels.cuda()
        return emb1, emb2, labels

    embedding, (train_set, dev_set, test_set) = data.load_dataset(config.dataset)
    model = mod.VDPWIModel(embedding, config)
    if config.restore:
        model.load(config.input_file)
    if not config.cpu:
        model = model.cuda()

    train_loader = utils.data.DataLoader(train_set, shuffle=True, batch_size=1, collate_fn=collate_fn)
    dev_loader = utils.data.DataLoader(dev_set, batch_size=1, collate_fn=collate_fn)
    test_loader = utils.data.DataLoader(test_set, batch_size=1, collate_fn=collate_fn)

    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = optim.RMSprop(params, lr=config.lr, alpha=config.decay, momentum=config.momentum)
    criterion = nn.KLDivLoss()
    return Context(model, train_loader, dev_loader, test_loader, optimizer, criterion, params)

def test(config):
    pass

def evaluate(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []
    for sent1, sent2, label_pmf in data_loader:
        scores = model(sent1, sent2)
        scores = F.softmax(scores).cpu().data.numpy()[0]
        prediction = np.dot(np.arange(1, len(scores) + 1), scores)
        truth = np.dot(np.arange(1, len(scores) + 1), label_pmf.cpu().data.numpy()[0])
        predictions.append(prediction); true_labels.append(truth)
    return EvaluateResult(stats.pearsonr(predictions, true_labels)[0], stats.spearmanr(predictions, true_labels)[0])

def train(config):
    context = create_context(config)
    for epoch_no in range(config.n_epochs):
        print("Epoch number: {}".format(epoch_no + 1))
        loader_wrapper = tqdm(enumerate(context.train_loader), total=len(context.train_loader), desc="Loss")
        context.model.train()
        for i, (sent1, sent2, label_pmf) in loader_wrapper:
            context.optimizer.zero_grad()
            scores = F.log_softmax(context.model(sent1, sent2))

            loss = context.criterion(scores, label_pmf)
            loss.backward()
            nn.utils.clip_grad_norm(context.params, 50)
            loader_wrapper.set_description("Loss = {}".format(loss.cpu().data[0]))
            context.optimizer.step()
        result = evaluate(context.model, context.dev_loader)
        print(result)

def main():
    config = data.Configs.base_config()
    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)

if __name__ == "__main__":
    main()