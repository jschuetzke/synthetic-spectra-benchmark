#!/bin/sh
python -m wandb login $WANDBKEY
python challenge_modifications.py
