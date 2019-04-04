import os

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from datasets.processors.bert_processor import convert_examples_to_features, accuracy
from utils.tokenization4bert import BertTokenizer


class BertEvaluator(object):
    def __init__(self, model, processor, args):
        self.args = args
        self.model = model
        self.processor = processor
        self.eval_examples = self.processor.get_dev_examples(args.data_dir)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    def evaluate(self):
        label_list = self.processor.get_labels()
        eval_features = convert_examples_to_features(self.eval_examples,
                                                     label_list,
                                                     self.args.max_seq_length,
                                                     self.tokenizer)

        print("Num. of examples =", len(self.eval_examples))
        print("Batch size = %d", self.args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        self.model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.args.device)
            input_mask = input_mask.to(self.args.device)
            segment_ids = segment_ids.to(self.args.device)
            label_ids = label_ids.to(self.args.device)

            with torch.no_grad():
                tmp_eval_loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                logits = self.model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy}

        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            print("***** Eval results *****")
            for key in sorted(result.keys()):
                print("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
