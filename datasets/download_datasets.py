import os
import csv
import functools
import logging
import importlib

# import torchtext.datasets.text_classification
from torchtext.datasets import DATASETS, URLS

DATASET_NAMES = ["AG_NEWS", "DBpedia", "IMDB", "SogouNews", "YahooAnswers", "YelpReviewPolarity"]

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
PATH_ROOT = ".local_data"
#
#
# def _print_skip_dataset(path):
#     print(f"Directory {path} exists, skipping download")
#
#
# def inject_path(dirname):
#     def decorator(func):
#         if not os.path.exists(PATH_ROOT):
#             os.mkdir(PATH_ROOT)
#         print(f"Downloading dataset {AG_NEWS}")
#         return lambda: func(path)
#         # else:
#         #     return lambda: print(f"Directory {path} exists, skipping download")
#
#     return decorator


# @inject_path(dirname="ag_news")
# def get_ag_news():
#     train_iter, test_iter = AG_NEWS(root=PATH_ROOT)
#     for _ in train_iter:
#         pass
#
#     for _ in test_iter:
#         pass
#
#     for filename in ("test.csv", "train.csv"):
#         with open(f"{path}/ag_news_csv/{filename}", "r") as f_in:
#             reader = csv.reader(f_in)
#             with open(f"{path}/{filename}", "w") as f_out:
#                 writer = csv.writer(f_out)
#                 for row in reader:
#                     # concatenate title and description rows
#                     new_row = [row[0], " ".join([row[1], row[2]])]
#                     writer.writerow(new_row)


# def get_sogou_news():
#     train_iter, test_iter = SogouNews(root=PATH_ROOT)
#     for _ in train_iter:
#         pass
#
#     for _ in test_iter:
#         pass

if __name__ == "__main__":
    for name in DATASET_NAMES:
        print(f"Downloading {name}")
        try:
            train_iter, test_iter = DATASETS[name](root=PATH_ROOT)
            for _ in train_iter:
                pass
            for _ in test_iter:
                pass

        except RuntimeError as e:
            if "content-disposition" in str(e):
                print(f"Cannot download dataset {name} due to Google Drive issue. Download it manually at",
                      URLS[name])
            else:
                print(e)
