import os
import csv

PATH_ROOT = ".local_data"


def process_ag_news():
    for filename in ("test.csv", "train.csv"):
        path_prefix = os.path.join(PATH_ROOT, "AG_NEWS")
        split_path_new = os.path.join(path_prefix, filename)
        split_path_old = os.path.join(path_prefix, f"temp_{filename}")
        os.rename(split_path_new, split_path_old)
        with open(split_path_old, "r") as f_in:
            reader = csv.reader(f_in)
            with open(split_path_new, "w+") as f_out:
                writer = csv.writer(f_out)
                for row in reader:
                    # concatenate title and description rows
                    new_row = [row[0], " ".join([row[1], row[2]])]
                    writer.writerow(new_row)

        os.remove(split_path_old)


def process_imdb():
    base_dir = os.path.join(PATH_ROOT, "IMDB/aclImdb_v1")
    train_dir, test_dir = [os.path.join(base_dir, split) for split in ("train", "test")]

    for split_dir in (train_dir, test_dir):
        neg, pos = [os.path.join(split_dir, label) for label in ("neg", "pos")]
        outfile = os.path.join(base_dir, os.path.basename(split_dir) + ".csv")
        with open(neg) as f_neg, open(pos) as f_pos, open(outfile, "w+") as f_out:
            # 0 = neg, 1 = pos
            for line in f_neg:
                f_out.write(f"01,{line}")
            for line in f_pos:
                f_out.write(f"10,{line}")


if __name__ == "__main__":
    process_ag_news()
    # process_imdb()
