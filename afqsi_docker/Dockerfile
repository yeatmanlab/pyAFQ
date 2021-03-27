###############################################################################
# Dockerfile to build qsiprep and pyAFQ 
###############################################################################

# Use QSIprep base image
FROM pennbbl/qsiprep:0.13.0RC1

# Install python dependencies
RUN \
    && pip install --no-cache-dir --ignore-installed git+https://github.com/yeatmanlab/pyAFQ.git@${COMMIT}\
    && pip install --no-cache-dir --ignore-installed git+https://github.com/bids-standard/pybids.git@0.9.3\
    && pip install fslpy\
    && pip install h5py
