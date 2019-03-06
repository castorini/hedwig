import torch
import torch.nn.functional as F

from .evaluator import Evaluator

def URL_maxF1_eval(predict_result, test_data_label):
    test_data_label = [item >= 1 for item in test_data_label]
    counter = 0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0

    for i, t in enumerate(predict_result):

        if t > 0.5:
            guess = True
        else:
            guess = False
        label = test_data_label[i]
        # print guess, label
        if guess == True and label == False:
            fp += 1.0
        elif guess == False and label == True:
            fn += 1.0
        elif guess == True and label == True:
            tp += 1.0
        elif guess == False and label == False:
            tn += 1.0
        if label == guess:
            counter += 1.0

    try:
        P = tp / (tp + fp)
        R = tp / (tp + fn)
        F = 2 * P * R / (P + R)
    except:
        P = 0
        R = 0
        F = 0

    accuracy = counter / len(predict_result)

    maxF1 = 0
    P_maxF1 = 0
    R_maxF1 = 0
    probs = predict_result
    sortedindex = sorted(range(len(probs)), key=probs.__getitem__)
    sortedindex.reverse()

    truepos = 0
    falsepos = 0
    for sortedi in sortedindex:
        if test_data_label[sortedi] == True:
            truepos += 1
        elif test_data_label[sortedi] == False:
            falsepos += 1
        precision = 0
        if truepos + falsepos > 0:
            precision = truepos / (truepos + falsepos)

        if (tp + fn) > 0:
            recall = truepos / (tp + fn)
        else:
            recall = 0
        f1 = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > maxF1:
                # print probs[sortedi]
                maxF1 = f1
                P_maxF1 = precision
                R_maxF1 = recall
    # print("PRECISION: {}, RECALL: {}, max_F1: {}".format(P_maxF1, R_maxF1, maxF1))
    return (accuracy, maxF1)

class PIT2015Evaluator(Evaluator):

    def get_scores(self):
        #self.model.eval()
        test_loss = 0
        true_labels = []
        predictions = []
        
        with torch.no_grad():
            for batch in self.data_loader:
                # Select embedding
                sent1, sent2 = self.get_sentence_embeddings(batch)

                output = self.model(sent1, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw)

                test_loss += F.nll_loss(output, batch.label, size_average=False).item()

                true_labels.extend(batch.label.detach().cpu().numpy())
                predictions.extend(output.detach().exp()[:, 1].cpu().numpy())

                del output

            test_loss /= len(batch.dataset.examples)
            accuracy, maxF1 = URL_maxF1_eval(predictions, true_labels)

        return [accuracy, test_loss, maxF1], ['accuracy', 'NLL loss', 'f1']