#!/bin/bash
LOG_BASE = "logs/profile_"
for i in {0..10};do
    LOG_FILE = $LOG_BASE + $i + ".log"
    python -u seedseq_profile.py --seq_id $i --seq_path "seed_sequence.json" --wbits 4 --load "esmfold_rtn_quant.pt" > LOG_FILE
done