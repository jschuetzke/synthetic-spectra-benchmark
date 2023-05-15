#!/bin/sh
python -m wandb login $WANDBKEY
python validate_hidden_activation.py
