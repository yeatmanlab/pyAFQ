ARG NVIDIAVERSION=12.0.1-devel-ubuntu20.04
FROM nvidia/cuda:${NVIDIAVERSION}

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# upgrade
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list
RUN apt update && \
    apt install --assume-yes apt-transport-https \
    	ca-certificates gnupg \
	software-properties-common gcc git wget numactl
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    	      | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ focal main"
RUN apt install -y cmake libncurses5-dev libtinfo6

# Anaconda
RUN wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
RUN bash /tmp/Anaconda3-2022.10-Linux-x86_64.sh -b -p /opt/anaconda
RUN rm -rf /tmp/Anaconda3-2022.10-Linux-x86_64.sh
RUN cd /opt && eval "$(/opt/anaconda/bin/conda shell.bash hook)"
ENV PATH /opt/anaconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/anaconda/lib:${LD_LIBRARY_PATH}

# python prereqs
RUN pip install numpy scipy cython nibabel dipy tqdm fslpy

# clone pyAFQ GPUStreamlines
RUN git clone https://github.com/yeatmanlab/pyAFQ.git /opt/pyAFQ
RUN cd /opt/pyAFQ && git reset --hard ${COMMIT}
RUN git clone --recursive  -b csaodf https://github.com/dipy/GPUStreamlines /opt/GPUStreamlines

# compile
RUN cd /opt/GPUStreamlines && mkdir build && cd build \
    &&  cmake -DCMAKE_INSTALL_PREFIX=/opt/GPUStreamlines/build/ \
      	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CXX_COMPILER=g++ \
	-DPYTHON_EXECUTABLE=$(which python) \
	.. \
    && make && make install

# install cuslines as package
RUN echo -e "from setuptools import setup, find_packages\nsetup(name='cuslines',version='0.0.1',packages=find_packages())" >> /opt/GPUStreamlines/build/cuslines/setup.py
RUN cd /opt/GPUStreamlines/build/cuslines && pip install -e .

# Install pyAFQ
RUN cd /opt/pyAFQ && pip install -e .
RUN /opt/pyAFQ/bin/pyAFQ download

ENTRYPOINT ["/opt/pyAFQ/bin/pyAFQ"]
