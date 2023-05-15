FROM tensorflow/tensorflow:2.9.1-gpu

WORKDIR app

ARG APIKEY_WANDB
ENV WANDBKEY=$APIKEY_WANDB

RUN pip install --no-cache-dir wandb

COPY x_train.npy x_val.npy x_test.npy y_train.npy y_val.npy y_test.npy ./
ADD model_implementations/ ./model_implementations/
COPY benchmark.py benchmark.sh ./
RUN chmod +x ./benchmark.sh

VOLUME /model_weights

CMD ["./benchmark.sh"]