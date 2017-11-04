#!/usr/bin/env bash

VERSION=9.0.5

wget https://github.com/usnistgov/trec_eval/archive/v${VERSION}.tar.gz
tar -xvzf v${VERSION}.tar.gz
cd trec_eval-${VERSION}
make
cd ..

rm -rf v${VERSION}.tar.gz
