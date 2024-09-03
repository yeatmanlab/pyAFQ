ARG NVIDIAVERSION=12.0.1-devel-ubuntu20.04
FROM nvidia/cuda:${NVIDIAVERSION}

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# upgrade
RUN apt-get update && apt-get install --assume-yes apt-transport-https \
	ca-certificates gnupg software-properties-common gcc git wget numactl

# Anaconda
RUN wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
RUN bash /tmp/Anaconda3-2022.10-Linux-x86_64.sh -b -p /opt/anaconda
RUN rm -rf /tmp/Anaconda3-2022.10-Linux-x86_64.sh
RUN cd /opt && eval "$(/opt/anaconda/bin/conda shell.bash hook)"
ENV PATH /opt/anaconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/anaconda/lib:${LD_LIBRARY_PATH}

# python prereqs
RUN pip install numpy scipy cython nibabel dipy tqdm fslpy
RUN pip install git+htttps://github.com/36000/GPUStreamlines.git@csd

# clone and install pyAFQ
RUN git clone https://github.com/yeatmanlab/pyAFQ.git /opt/pyAFQ
RUN cd /opt/pyAFQ && git reset --hard ${COMMIT}
RUN cd /opt/pyAFQ && pip install -e .
RUN /opt/pyAFQ/bin/pyAFQ download

ENTRYPOINT ["/opt/pyAFQ/bin/pyAFQ"]
