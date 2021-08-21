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
# COPY init.sh /tmp

# SHELL ["/bin/bash", "--login", "-c"]
# RUN conda init bash
# RUN cat /tmp/init.sh >> ~/.bashrc

# RUN echo >> ~/.bashrc "__conda_setup=\"$('conda' 'shell.bash' 'hook' 2> /dev/null)\"" \
    # && echo >> ~/.bashrc "eval \"\$__conda_setup\"" \
    # && echo >> ~/.bashrc "unset __conda_setup"

# RUN conda --version

# RUN apt-get update --fix-missing && apt-get install -y \
#     cmake \
#     python2.7 \
#     python-pip \
#     python3.8 \
#     python3-pip \
#     hmmer \
#     hhsuite \
#     git \
#     gfortran \
#     liblapack-dev \
#     libopenblas-dev
#
#
# WORKDIR /app
# COPY requirements_alignment.txt /tmp
# RUN pip3 install -r /tmp/requirements_alignment.txt
# COPY requirements_deepsequence.txt /tmp
# RUN pip2 install -r /tmp/requirements_deepsequence.txt
#
# # Clone libgpuarray repo and move into it
# RUN cd /root && git clone https://github.com/Theano/libgpuarray.git && cd libgpuarray && \
# # Make and move into build directory
#   mkdir Build && cd Build && \
# # CMake
#   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr && \
# # Make
#   make -j"$(nproc)" && \
#   make install
#
# RUN cd /root/libgpuarray && pip2 install Cython MarkupSafe && python2 setup.py build && python2 setup.py install
#
COPY DeepSequence /app
COPY examples/run_svi.py /app
COPY examples/predict_single_mutant.py /app
COPY align.py /app
COPY run_all.sh /app

ENTRYPOINT ["/app/run_all.sh"]
