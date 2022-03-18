import os

PATH_ROOT = ".local_data"


def process_imdb_dataset():
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
    process_imdb_dataset()
