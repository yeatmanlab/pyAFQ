###############################################################################
# Dockerfile to build pyAFQ 
###############################################################################

# Use python base image
FROM python:3.8

# Install pyAFQ
RUN \
    && pip install --no-cache-dir git+https://github.com/yeatmanlab/pyAFQ.git@${COMMIT}\
    && pip install fslpy\
    && pip install h5py
