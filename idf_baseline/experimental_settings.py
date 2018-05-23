# The following 5 conditions vary
# idf_source, stopwords_and_stemming, punctuation, words-with-hyphens
import argparse
import itertools
import shlex
import subprocess

class Setting(object):
    def __init__(self, label, value_flag_map):
        self.label = label
        self.choice_flags = value_flag_map
    
    def get_settings(self):
        return self.choice_flags.keys()

    def get_choice(self, setting):
        return self.choice_flags[setting]

    def get_options(self):
        options = []
        for key in self.choice_flags.keys():
            options.append("{}:{}".format(self.label, key))
        return options

class Experiments(object):
    def __init__(self, qa_dataset):
        self.settings = {}
        self.combinations = []
        self.qa_data = qa_dataset
        self.cmd_root = "python qa-data-only-idf.py {} run".format(self.qa_data)
        self.eval_cmd_root = "../../Anserini/eval/trec_eval.9.0/trec_eval -m map -m recip_rank -m bpref"
        self.rbp_cmd_root = "rbp_eval"

    def add_setting(self, setting):
        self.settings[setting.label] = setting
        self._setup_combinations()


    def _setup_combinations(self):
        all_settings = []
        for setting in self.settings.values():
            all_settings.append(setting.get_options())

        self.combinations = []
        for c in itertools.product(*all_settings):
             self.combinations.append(c)

    def _run_cmd(self, cmd):
        pargs = shlex.split(cmd)
        p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, \
                                bufsize=1, universal_newlines=True)
        pout, perr = p.communicate()
        return pout, perr
    
    def _run_eval(self):
        for split in ['train-all', 'raw-dev', 'raw-test']:
            cmd = '{} {}/{}.qrel run.{}.idfsim'.format(self.eval_cmd_root,
                                                       self.qa_data, split, split)
            out, err = self._run_cmd(cmd)
            print(split, '------' )
            # trec_eval scores
            metrics = []
            scores = []
            for line in str(out).split('\n'):
                if not line.strip().split():
                    continue
                fields = line.strip().split()
                metrics.append(fields[0])
                scores.append(fields[-1])
            
            # rbp_eval scores
            cmd = '{} {}/{}.qrel run.{}.idfsim'.format(self.rbp_cmd_root,
                                                       self.qa_data, split, split)
            out, err = self._run_cmd(cmd)
            for line in str(out).split('\n'):
                if not line.startswith('p= 0.50'):
                    continue
                metrics.append('rbp_p0.5')
                scores.append(' '.join(line.strip().split()[-2:]))
            
            print('\t'.join(metrics))
            print('\t'.join(scores))


    def run(self, indices):
        """
        runs a particular combination of settings
        """
        for ci in indices:
            combo = self.combinations[ci]
            print(combo)
            cmd_args = []
            for setting_choice in combo:
                setting, choice = setting_choice.split(':')
                cmd_args.append(self.settings[setting].choice_flags[choice])
            cmd = '{} {}'.format(self.cmd_root, ' '.join(cmd_args))
            print(cmd)
            out, err = self._run_cmd(cmd)
            self._run_eval()

    def run_all(self):
        """
        runs all experiments
        """
        pass
    
    def list_settings(self):
        """
        lists all settings
        """
        for c in enumerate(self.combinations):
            print(c)
        print("--run X Y Z to run combinations number X Y Z")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Lists exerimental settings and runs experiments")
    ap.add_argument("--list", help="lists all available experimental settings combinations",
                    action="store_true")
    ap.add_argument("--run", help='runs experimenal setting combination NUMBER(s)', 
                    nargs="+", type=int)
    ap.add_argument("--runall", help="runs all experiments in order", action="store_true")
    ap.add_argument("index_path", help="required for some combination of experiments")
    ap.add_argument('qa_data', help="path to the QA dataset",
                    choices=['../../Castor-data/TrecQA', '../../Castor-data/WikiQA'])

    args = ap.parse_args()

    experiments = Experiments(args.qa_data)

    experiments.add_setting(Setting('idf_source', {
        'qa-data':'',
        'corpus-index': '--index-for-corpusIDF {}'.format(args.index_path)
    }))

    experiments.add_setting(Setting('stop_stem', {
        'yes':'--stop-and-stem',
        'no': ''
    }))

    experiments.add_setting(Setting('punctuation', {
        'keep': '',
        'remove': '--stop-punct'
    }))

    experiments.add_setting(Setting('dash_words', {
        'keep': '',
        'split': '--dash-split'
    }))

    experiments.list_settings()

    if args.run:
        experiments.run(args.run)
    
    if args.runall:
        experiments.run_all()
