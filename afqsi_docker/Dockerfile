###############################################################################
# Dockerfile to build qsiprep and pyAFQ 
###############################################################################

# Use QSIprep base image
FROM pennbbl/qsiprep:0.13.0RC1

RUN echo ${FREESURFER_LICENSE} > ${FREESURFER_HOME}/license.txt
ENV FS_LICENSE=${FREESURFER_HOME}/license.txt \
    DUECREDIT_ALLOW_FAIL=yes
ENTRYPOINT ["/usr/local/miniconda/bin/qsiprep"]

# Install python dependencies
RUN \
    && pip install --no-cache-dir git+https://github.com/yeatmanlab/pyAFQ.git@${COMMIT}\
    && pip install --no-cache-dir git+https://github.com/bids-standard/pybids.git@0.9.3
    && pip install fslpy
    && pip install h5py
