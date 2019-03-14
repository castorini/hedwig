#!/bin/sh
python . --clip_norm 50 --decay 0.95 --lr 1E-4 --mbatch_size 1 --momentum 0 --optimizer rmsprop --weight_decay 0
