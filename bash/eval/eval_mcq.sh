#!/bin/bash

model_list=`find $1 -maxdepth 1 -type d  | sort -t'/' -k2.2r -k2.1`
# KSHOTS=$2

for KSHOTS in 0 5 10 20;
do
    echo $KSHOTS
    for f in $model_list; do
        if echo "$f" | grep -q "_hf"; then
            echo $f
            python scripts/eval/evaluate_on_mcq.py $f TLLM/eval-MCQ-ceval-hanlin $KSHOTS --name $f --version "kshots=$KSHOTS"
        fi
    done
done