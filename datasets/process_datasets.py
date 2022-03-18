import csv
import os
import sys

PATH_ROOT = ".local_data"

csv.field_size_limit(sys.maxsize)


def concat_text(old_dir_prefix, new_dir_prefix):
    """Concatenates all non-label entries in each dataset row and saves new csv file"""
    for filename in ("test.csv", "train.csv"):
        split_path_old = os.path.join(old_dir_prefix, filename)
        split_path_temp = os.path.join(old_dir_prefix, f"temp_{filename}")
        split_path_new = os.path.join(new_dir_prefix, filename)

        os.rename(split_path_old, split_path_temp)
        with open(split_path_temp, "r") as f_in:
            reader = csv.reader(f_in)
            with open(split_path_new, "w+") as f_out:
                writer = csv.writer(f_out)
                for row in reader:
                    # concatenate title and description rows
                    new_row = [row[0], " ".join(row[1:])]
                    writer.writerow(new_row)


def process_ag_news():
    path_prefix = os.path.join(PATH_ROOT, "AG_NEWS")
    concat_text(path_prefix, path_prefix)


def process_dbpedia():
    path_prefix = os.path.join(PATH_ROOT, "DBpedia", "dbpedia_csv")
    new_path_prefix = os.path.join(PATH_ROOT, "DBpedia")
    concat_text(path_prefix, new_path_prefix)


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


def process_sogou_news():
    path_prefix = os.path.join(PATH_ROOT, "SogouNews", "sogou_news_csv")
    new_path_prefix = os.path.join(PATH_ROOT, "SogouNews")
    concat_text(path_prefix, new_path_prefix)


def process_yahoo_answers():
    path_prefix = os.path.join(PATH_ROOT, "YahooAnswers", "yahoo_answers_csv")
    new_path_prefix = os.path.join(PATH_ROOT, "YahooAnswers")
    concat_text(path_prefix, new_path_prefix)


def process_yelp_review_polarity():
    path_prefix = os.path.join(PATH_ROOT, "YelpReviewPolarity", "yelp_review_polarity_csv")
    new_path_prefix = os.path.join(PATH_ROOT, "YelpReviewPolarity")
    concat_text(path_prefix, new_path_prefix)


if __name__ == "__main__":
    process_ag_news()
    process_imdb()
    process_dbpedia()
    process_sogou_news()
    process_yahoo_answers()
    process_yelp_review_polarity()
