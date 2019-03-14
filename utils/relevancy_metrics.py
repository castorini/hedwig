import os
import subprocess
import time


def get_map_mrr(qids, predictions, labels, device=0, keep_results=False):
    """
    Get the map and mrr using the trec_eval utility.
    qids, predictions, labels should have the same length.
    device is not a required parameter, it is only used to prevent potential naming conflicts when you
    are calling this concurrently from different threads of execution.
    :param qids: query ids of predictions and labels
    :param predictions: iterable of predictions made by the models
    :param labels: iterable of labels of the dataset
    :param device: device (GPU index or -1 for CPU) for identification purposes only
    """
    qrel_fname = 'trecqa_{}_{}.qrel'.format(time.time(), device)
    results_fname = 'trecqa_{}_{}.results'.format(time.time(), device)
    qrel_template = '{qid} 0 {docno} {rel}\n'
    results_template = '{qid} 0 {docno} 0 {sim} castor-model\n'
    with open(qrel_fname, 'w') as f1, open(results_fname, 'w') as f2:
        docnos = range(len(qids))
        for qid, docno, predicted, actual in zip(qids, docnos, predictions, labels):
            f1.write(qrel_template.format(qid=qid, docno=docno, rel=actual))
            f2.write(results_template.format(qid=qid, docno=docno, sim=predicted))

    trec_eval_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trec_eval-9.0.5/trec_eval')
    trec_out = subprocess.check_output([trec_eval_path, '-m', 'map', '-m', 'recip_rank', qrel_fname, results_fname])
    trec_out_lines = str(trec_out, 'utf-8').split('\n')
    mean_average_precision = float(trec_out_lines[0].split('\t')[-1])
    mean_reciprocal_rank = float(trec_out_lines[1].split('\t')[-1])

    if keep_results:
        print("Saving prediction file to {}".format(results_fname))
        print("Saving qrel file to {}".format(qrel_fname))
    else:
        os.remove(results_fname)
        os.remove(qrel_fname)

    return mean_average_precision, mean_reciprocal_rank
