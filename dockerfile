FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN conda update -y -n base -c defaults conda
RUN conda install -y -c conda-forge opencv scikit-video
RUN conda install -y tensorboard scikit-learn


COPY inference.py .
COPY training.py .
COPY conversion_utils.py .
COPY sshd_config /etc/ssh/sshd_config
COPY training_parallel.py .
COPY run.sh .
COPY dataset dataset
COPY model model
COPY training_lstm.py .