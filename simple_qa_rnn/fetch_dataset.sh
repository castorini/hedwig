#!/bin/bash

# download dataset and put it in data directory
mkdir data
pushd data
wget https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz
tar -xvzf SimpleQuestions_v2.tgz
popd