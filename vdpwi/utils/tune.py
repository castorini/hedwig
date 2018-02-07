import os
import random

class RandomParamIterator(object):
    def __init__(self, param_sets):
        self.param_sets = param_sets

    def random_param_set(self):
        param_set = {}
        for param_key, param_values in self.param_sets.items():
            param_set[param_key] = random.choice(param_values)
        return param_set

class Tuner(object):
    def __init__(self, *iterators, limit=500):
        self.iterators = iterators
        self.limit = limit

    def start(self):
        for i in range(self.limit):
            iterator = random.choice(self.iterators)
            params = iterator.random_param_set()
            print(params)
            arg_str = " ".join("--{}={}".format(k, v) for k, v in params.items())
            os.system("python . {} --output_file local_saves/model{}.pt".format(arg_str, i))

def main():
    vgg_param_sets = dict(classifer=["vdpwi"], decay=[0.99, 0.95], lr=[5E-4, 1E-4], mbatch_size=[8, 16], 
        optimizer=["adam", "rmsprop"], weight_decay=[0, 1E-5, 5E-4], momentum=[0, 0.15, 0.05])
    res_param_sets = dict(classifier=["resnet"], clip_norm=[5, 7, 9], decay=[0.9, 0.95], lr=[5E-3, 1E-3, 5E-4],
        mbatch_size=[8, 16, 32], rnn_hidden_dim=[150, 250, 300], res_fmaps=[16, 24, 32], res_layers=[4, 8, 16, 24])
    vgg_iterator = RandomParamIterator(vgg_param_sets)
    res_iterator = RandomParamIterator(res_param_sets)
    Tuner(vgg_iterator).start()

if __name__ == "__main__":
    main()