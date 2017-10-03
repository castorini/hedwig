import shlex
import subprocess

def evaluate(instances, valid, config):
    sorted_instances = sorted(instances, key=lambda x: (x[0]))
    with open('{}.{}.run.txt'.format(valid, config), 'w') as run, open('{}.{}.qrel.txt'.format(valid, config), 'w') as qrel:
        i = 0
        for instance in sorted_instances:
            qid, predicted, score, gold = instance[0], instance[1], instance[2], instance[3]

            # 32.1 0 1 0 0.13309887051582336 smmodel
            run.write('{} 0 {} 0 {} sm_model\n'.format(qid, i, score))
            qrel.write('{} 0 {} {}\n'.format(qid, i, gold))
            i += 1

    pargs = shlex.split("./eval/trec_eval.9.0/trec_eval -m map -m recip_rank {}.{}.qrel.txt {}.{}.run.txt"
                        .format(valid, config, valid, config))
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pout, perr = p.communicate()
    lines = pout.split(b'\n')
    map = float(lines[0].strip().split()[-1])
    mrr = float(lines[1].strip().split()[-1])
    return map, mrr