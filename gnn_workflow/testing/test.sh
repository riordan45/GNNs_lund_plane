#!/bin/bash
index=$1
cd /eos/user/j/jzahredd/JetTagging/
source miniconda3/bin/activate
conda activate rootenv
cd /afs/cern.ch/user/r/riordan/private/jobs/workflow
python3 test.py $index
