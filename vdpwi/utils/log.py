import datetime
import sys

from tensorboardX import SummaryWriter

class LogWriter(object):
    def __init__(self, run_name_fmt="run_{}"):
        self.writer = SummaryWriter()
        self.run_name = run_name_fmt.format(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        self.train_idx = 0
        self.dev_idx = 0

    def log_hyperparams(self):
        self.writer.add_text("{}/hyperparams".format(self.run_name), " ".join(sys.argv))

    def log_train_loss(self, loss):
        self.writer.add_scalar("{}/train_loss".format(self.run_name), loss, self.train_idx)
        self.train_idx += 1

    def log_dev_metrics(self, pearsonr, spearmanr):
        results = dict(pearsonr=pearsonr, spearmanr=spearmanr)
        self.writer.add_scalars("{}/dev_metrics".format(self.run_name), results, self.dev_idx)
        self.dev_idx += 1

    def next(self):
        self.i += 1
