#!/bin/sh
mkdir -p data
mkdir -p saves
wget http://ocp59jkku.bkt.clouddn.com/sst-1.zip -P data/
wget http://ocp59jkku.bkt.clouddn.com/sst-2.zip -P data/
unzip data/sst-1.zip -d data/
unzip data/sst-2.zip -d data/
