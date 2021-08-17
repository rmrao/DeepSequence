FROM nvidia/cuda:10.2-base

RUN apt-get update --fix-missing && apt-get install -y \
    python2.7 \
    python-pip \
    python3.8 \
    python3-pip \
    hmmer \
    hhsuite \
    git

WORKDIR /app
COPY requirements_alignment.txt /tmp
RUN pip3 install -r /tmp/requirements_alignment.txt
COPY requirements_deepsequence.txt /tmp
RUN pip2 install -r /tmp/requirements_deepsequence.txt

COPY DeepSequence /app
COPY examples/run_svi.py /app
COPY examples/predict_single_mutant.py /app
COPY align.py /app
COPY run_all.py /app

ENTRYPOINT ["/app/run_all.py"]
