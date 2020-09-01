FROM python:3.7

# Install libgl, xvfb 
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y xvfb

# Install pyAFQ
RUN python -m pip install --upgrade pip && \
    pip install pyafq[dev,fury,plotly]

# ENTRYPOINT [""]