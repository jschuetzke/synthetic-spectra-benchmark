#!/bin/sh
python -m wandb login $WANDBKEY
python benchmark.py
