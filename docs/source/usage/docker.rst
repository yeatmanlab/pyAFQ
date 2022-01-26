The pyAFQ docker image
~~~~~~~~~~~~~~~~~~~~~~

Everytime a new commit is made to master the
`pyAFQ github <https://github.com/yeatmanlab/pyAFQ>`_,
a new image is pushed to the
`NRDG github <https://github.com/orgs/nrdg/packages/container/package/pyafq>`_.
This image contains an installation of the latest version of
pyAFQ with fslpy and h5py. This image also contains an entrypoint, and can be
run with::
    docker run -v bids_dir:/bids_dir:rw ghcr.io/nrdg/pyafq bids_dir/config.toml
More information on config.toml can be found under "The pyAFQ configuration file".
