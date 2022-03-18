import os
import csv
from torchtext.datasets import AG_NEWS

if __name__ == "__main__":
    path_root = ".local_data/ag_news"
    if not os.path.exists(path_root):
        os.mkdir(path_root)

    train_iter, test_iter = AG_NEWS(root=path_root)

    for filename in ("test.csv", "train.csv"):
        with open(f"{path_root}/ag_news_csv/{filename}", "r") as f_in:
            reader = csv.reader(f_in)
            with open(f"{path_root}/{filename}", "w") as f_out:
                writer = csv.writer(f_out)
                for row in reader:
                    # concatenate title and description rows
                    new_row = [row[0], " ".join([row[1], row[2]])]
                    writer.writerow(new_row)
