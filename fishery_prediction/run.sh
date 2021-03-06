#!/bin/bash

set -e

# ./train.py wu day
# ./train.py wu week
# ./train.py chi day
# ./train.py chi week

# ./predict.py wu day wu_day_large.h5
./predict.py wu week wu_week_large_11.h5
./predict.py chi day chi_day_large_41.h5
./predict.py chi week chi_week_large_21.h5
echo 'all done'

# echo '1. exp1: Whether to drop data before 2013' >> logs/exp.txt
# ./train.py 1 | tail -n 30 >> logs/exp.txt
# echo '-------------------------------------------' >> logs/exp.txt
# echo '' >> logs/exp.txt
# echo '2. exp2: backward selection' >> logs/exp.txt
# ./train.py 2 | tail -n 30 >> logs/exp.txt
# echo '-------------------------------------------' >> logs/exp.txt
# echo '' >> logs/exp.txt

# ./preprocess.py 1 7 1
# ./train.py base 7d > logs/base_7d.txt
# ./train.py large 7d > logs/large_7d.txt
#
# ./preprocess.py 1 30 1
# ./train.py base 30d > logs/base_30d.txt
# ./train.py large 30d > logs/large_30d.txt
#
# ./preprocess.py 7 4 1
# ./train.py base 4w > logs/base_4w.txt
# ./train.py large 4w > logs/large_4w.txt
#
# ./preprocess.py 7 8 1
# ./train.py base 8w > logs/base_8w.txt
# ./train.py large 8w > logs/large_8w.txt
#
# ./preprocess.py 7 24 1
# ./train.py base 24w > logs/base_24w.txt
# ./train.py large 24w > logs/large_24w.txt
#
# ./preprocess.py 30 6 1
# ./train.py base 6m > logs/base_6m.txt
# ./train.py large 6m > logs/large_6m.txt
