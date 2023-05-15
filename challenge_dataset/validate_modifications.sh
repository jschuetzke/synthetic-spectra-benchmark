#!/bin/sh
python -m wandb login $WANDBKEY
python validate_modifications.py
