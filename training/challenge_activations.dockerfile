FROM tensorflow/tensorflow:2.9.1-gpu

WORKDIR app

ARG APIKEY_WANDB
ENV WANDBKEY=$APIKEY_WANDB

RUN pip install --no-cache-dir wandb

COPY challenge_dataset/x_train.npy challenge_dataset/x_val.npy challenge_dataset/x_test.npy ./
COPY challenge_dataset/y_train.npy challenge_dataset/y_val.npy challenge_dataset/y_test.npy ./
ADD model_implementations/ ./model_implementations/
COPY training/challenge_activations.py ./
COPY training/challenge_activations.sh ./
RUN chmod +x ./challenge_activations.sh

CMD ["./challenge_activations.sh"]