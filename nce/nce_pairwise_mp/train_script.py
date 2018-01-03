import argparse
from random import uniform
from subprocess import call


# set parameters from random uniform sampling
def get_param():
    for eps in [1e-8, 1e-6]:
        for learning_rate in [0.0001, 0.0003, 0.0006, 0.001]:
            for reg in [3e-5, 9e-5, 3e-4, 9e-4]:
                yield learning_rate, eps, reg

def run(device, epochs, neg_sample, neg_num, dataset, dev_log_interval, batch_size):
    # for _ in range(count):
    param_gen = get_param()
    for learning_rate, eps, reg in param_gen:

        filename = "grid_{dataset}_lr_{learning_rate}_eps_{eps}_reg_{reg}_device_{dev}.txt".format(
            learning_rate=learning_rate, eps=eps, reg=reg, dev=device, dataset=dataset)
        model_name = filename[:-4] + ".castor"

        command = "python -u main.py saved_models/{model} --epochs {epo} --device {dev} --dataset {dataset} " \
                  "--batch-size {batch_size} --lr {learning_rate} --epsilon {eps} --regularization {reg} --tensorboard " \
                  "--run-label {label} --dev_log_interval {dev_log_interval} --neg_sample {neg_sample} --neg_num {neg_num}" \
            .format(epo=epochs, model=model_name, dev=device, dataset=dataset, batch_size=batch_size,
                    learning_rate=learning_rate, eps=eps,
                    reg=reg, label=filename, dev_log_interval=dev_log_interval, neg_sample=neg_sample, neg_num=neg_num)

        print("Running: " + command)
        with open(filename, 'w') as outfile:
            call(command, shell=True, stderr=outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper parameters sweeper')
    parser.add_argument('--device', type=int, default=2, help='GPU device, -1 for CPU (default: 0)')
    parser.add_argument('--epochs', type=int, default=8, help='number of epochs to run')
    parser.add_argument('--neg_num', type=int, default=8, help='number of negative samples')
    parser.add_argument('--neg_sample', type=str, default="max", help='strategy of negative samples')
    parser.add_argument('--dataset', type=str, default="trecqa", help='dataset')
    parser.add_argument('--dev_log_interval', type=int, default=150, help='number of negative samples')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    args = parser.parse_args()
    run(args.device, args.epochs, neg_sample=args.neg_sample, neg_num=args.neg_num, dataset=args.dataset,
        dev_log_interval=args.dev_log_interval, batch_size=args.batch_size)