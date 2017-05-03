#!/bin/csh -f

exp_dir=$1
judgement=${exp_dir}/gold.txt
output=${exp_dir}/submission.txt

./trec_eval-8.0/trec_eval -q -c ${judgement} ${output} > ${output}.treceval
tail -29 ${output}.treceval | grep -e 'map' -e 'recip_rank'
exit 0
