###############################################################################
# Dockerfile to build qsiprep and pyAFQ
###############################################################################

# Use QSIprep base image
FROM pennbbl/qsiprep:0.14.3

ARG COMMIT

# Install python dependencies
RUN pip install git+https://github.com/yeatmanlab/pyAFQ.git@${COMMIT}
RUN pip install pytest==6.0.1
