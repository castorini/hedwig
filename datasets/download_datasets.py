import logging

from torchtext.datasets import DATASETS, URLS

DATASET_NAMES = ["AG_NEWS", "DBpedia", "IMDB", "SogouNews", "YahooAnswers", "YelpReviewPolarity"]

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
PATH_ROOT = ".local_data"

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
