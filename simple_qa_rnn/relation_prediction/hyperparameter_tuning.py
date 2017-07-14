from random import randint, uniform
from subprocess import call

epochs = 50
count = 20
for id in range(count):
    learning_rate = 10 ** uniform(-5, -4)
    d_hidden = randint(550, 600)
    n_layers = randint(4, 5)
    dropout = uniform(0.5, 0.6)
    clip = uniform(0.6, 0.7)

    command = "python train.py --dev_every 500 --log_every 250 --batch_size 32 " \
                "--epochs {} --lr {} --d_hidden {} --n_layers {} --dropout_prob {} --clip_gradient {} >> " \
                    "results.txt".format(epochs, learning_rate, d_hidden, n_layers, dropout, clip)

    print("Running: " + command)
    call(command, shell=True)
