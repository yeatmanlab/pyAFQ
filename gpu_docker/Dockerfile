FROM nvcr.io/nvidia/cuda:11.0.3-devel

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# upgrade
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt update && \
    apt install --assume-yes apt-transport-https \
    	ca-certificates gnupg \
	software-properties-common gcc git wget numactl
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    	      | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ focal main"
RUN apt install -y cmake libncurses5-dev libtinfo6

# Anaconda
RUN cd /opt && wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh \
    && bash /tmp/Anaconda3-2022.10-Linux-x86_64.sh -b -p /opt/anaconda \
    && rm -rf /tmp/Anaconda3-2022.10-Linux-x86_64.sh \
    && eval "$(/opt/anaconda/bin/conda shell.bash hook)"
ENV PATH /opt/anaconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/anaconda/lib:${LD_LIBRARY_PATH}

# python prereqs
RUN pip install numpy scipy cython nibabel dipy tqdm fslpy h5py

# clone pyAFQ GPUStreamlines
RUN git clone https://github.com/yeatmanlab/pyAFQ.git@${COMMIT} /opt/pyAFQ
RUN git clone --recursive https://github.com/dipy/GPUStreamlines /opt/GPUStreamlines

# compile
RUN cd /opt/GPUStreamlines && mkdir build && cd build \
    &&  cmake -DCMAKE_INSTALL_PREFIX=/opt/GPUStreamlines/build/ \
      	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_CXX_COMPILER=g++ \
	-DPYTHON_EXECUTABLE=$(which python) \
	.. \
    && make && make install

# Install pyAFQ
RUN cd /opt/pyAFQ && pip install -e .

# Bring in cuslines
RUN cp -r /opt/GPUStreamlines/build/* /opt/pyAFQ/bin/

ENTRYPOINT ["/opt/pyAFQ/bin/pyAFQ"]
