###############################################################################
# Dockerfile to build qsiprep and pyAFQ 
###############################################################################

# Use QSIprep base image
FROM pennbbl/qsiprep:0.13.0RC1

ARG COMMIT

# Install python dependencies
RUN pip install --no-cache-dir --ignore-installed git+https://github.com/yeatmanlab/pyAFQ.git@${COMMIT}
RUN pip install --no-cache-dir --ignore-installed git+https://github.com/bids-standard/pybids.git@0.9.3
RUN pip install fslpy
RUN pip install h5py
RUN pip install pytest==6.0.1
RUN pip uninstall -y dataclasses
