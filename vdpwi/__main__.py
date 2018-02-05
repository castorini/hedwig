from collections import namedtuple

from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils

from utils.log import LogWriter
import data
import model as mod

Context = namedtuple("Context", "model, train_loader, dev_loader, test_loader, optimizer, criterion, params, log_writer")
EvaluateResult = namedtuple("EvaluateResult", "pearsonr, spearmanr")

def create_context(config):
    def collate_fn(batch):
        emb1 = []
        emb2 = []
        labels = []
        cmp_labels = []
        for s1, s2, l, cl in batch:
            emb1.append(s1)
            emb2.append(s2)
            labels.append(l)
            cmp_labels.append(cl)
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
        return emb1, emb2, labels, cmp_labels

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
    if config.optimizer == "adam":
        optimizer = optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == "rmsprop":
        optimizer = optim.RMSprop(params, lr=config.lr, alpha=config.decay, momentum=config.momentum, weight_decay=config.weight_decay)
    criterion = nn.KLDivLoss()
    log_writer = LogWriter()
    return Context(model, train_loader, dev_loader, test_loader, optimizer, criterion, params, log_writer)

def test(config):
    context = create_context(config)
    result = evaluate(context, context.test_loader)
    print("Final test result: {}".format(result))

def evaluate(context, data_loader):
    model = context.model
    model.eval()
    predictions = []
    true_labels = []
    for sent1, sent2, _, truth in data_loader:
        scores = model(sent1, sent2)
        scores = F.softmax(scores).cpu().data.numpy()[0]
        prediction = np.dot(np.arange(1, len(scores) + 1), scores)
        predictions.append(prediction); true_labels.append(truth[0][0])
    
    pearsonr = stats.pearsonr(predictions, true_labels)[0]
    spearmanr = stats.spearmanr(predictions, true_labels)[0]
    context.log_writer.log_dev_metrics(pearsonr, spearmanr)
    return EvaluateResult(pearsonr, spearmanr)

def train(config):
    context = create_context(config)
    context.log_writer.log_hyperparams()
    best_dev_pr = 0
    for epoch_no in range(config.n_epochs):
        print("Epoch number: {}".format(epoch_no + 1))
        loader_wrapper = tqdm(enumerate(context.train_loader), total=len(context.train_loader), desc="Loss")
        context.model.train()
        loss = 0
        for i, (sent1, sent2, label_pmf, _) in loader_wrapper:
            context.optimizer.zero_grad()
            scores = F.log_softmax(context.model(sent1, sent2))

            loss = context.criterion(scores, label_pmf) + loss
            if i % config.mbatch_size == (config.mbatch_size - 1):
                loss /= config.mbatch_size
                loss.backward()
                nn.utils.clip_grad_norm(context.params, config.clip_norm)
                context.optimizer.step()

                loss = loss.cpu().data[0]
                loader_wrapper.set_description("Loss: {:<8}".format(round(loss, 5)))
                context.log_writer.log_train_loss(loss)
                loss = 0
        result = evaluate(context, context.dev_loader)
        print("Dev result: {}".format(result))
        if best_dev_pr < result.pearsonr:
            best_dev_pr = result.pearsonr
            print("Saving best model...")
            context.model.save(config.output_file)

def main():
    config = data.Configs.base_config()
    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)

if __name__ == "__main__":
    main()