FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update && apt-get install -y \
    wget \
    hmmer \
    hhsuite \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -n alignment biopython scipy numpy
COPY env.yaml /tmp
RUN conda env create -f /tmp/env.yaml
RUN conda clean --all -y

COPY DeepSequence /app
COPY examples/run_svi.py /app
COPY examples/predict_single_mutant.py /app
COPY align.py /app
COPY run_all.sh /app

ENTRYPOINT ["/app/run_all.sh"]
