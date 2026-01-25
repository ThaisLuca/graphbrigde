#!/bin/sh

python3 side_tune_no_transfer.py --dataset="bace" --device=0
python3 side_tune_no_transfer.py --dataset="bbbp" --device=0
python3 side_tune_no_transfer.py --dataset="hiv" --device=0
python3 side_tune_no_transfer.py --dataset="clintoxapproved" --device=0
python3 side_tune_no_transfer.py --dataset="clintoxtoxicity" --device=0
python3 side_tune_no_transfer.py --dataset="zinc_standard_agent" --device=0
