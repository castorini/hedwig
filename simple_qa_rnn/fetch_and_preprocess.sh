#!/bin/bash

# download GloVe word embeddings
python2 scripts/download.py

glove_dir="data/glove"
glove_pre="glove.840B"
glove_dim="300d"
if [ ! -f $glove_dir/$glove_pre.$glove_dim.pt ]; then
    python scripts/convert_wordvecs.py $glove_dir/$glove_pre.$glove_dim.txt \
                                             $glove_dir/$glove_pre.$glove_dim.pt
else
	echo "The processed word embeddings file - $glove_dir/$glove_pre.$glove_dim.pt - already exists!"
fi