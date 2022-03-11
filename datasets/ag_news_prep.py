import os
import csv
from torchtext.datasets import AG_NEWS

if __name__ == "__main__":
    if not os.path.exists(".ag_news"):
        os.mkdir(".ag_news")

    # train_iter, test_iter = AG_NEWS(root=".ag_news")

    for filename in ("test.csv", "train.csv"):
        with open(f".ag_news/ag_news_csv/{filename}", "r") as f_in:
            reader = csv.reader(f_in)
            outfile = f".ag_news/{filename}"
            with open(f".ag_news/{filename}", "w") as f_out:
                writer = csv.writer(f_out)
                for row in reader:
                    # concatenate title and description rows
                    new_row = [row[0], " ".join([row[1], row[2]])]
                    writer.writerow(new_row)
