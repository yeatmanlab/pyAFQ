FROM python:3.7

# Install libgl, xvfb 
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y xvfb

# Download pyAFQ to install from source
RUN git clone -b 0.4 https://github.com/yeatmanlab/pyAFQ.git

# Install pyAFQ, remarshal
RUN python -m pip install --upgrade pip && \
    pip install ./pyAFQ"[dev,fury,plotly]" && \
    pip install remarshal==0.12.0

# Set entrypoint
RUN ["chmod", "+x", "/pyAFQ/bin/pyAFQ"]
RUN ["chmod", "+x", "/pyAFQ/bin/bl_entrypoint"]
ENTRYPOINT ["/pyAFQ/bin/bl_entrypoint"]
