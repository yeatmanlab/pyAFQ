import os.path as op


def drop_extension(path):
    base_fname = op.basename(path).split('.')[0]
    return path.split(base_fname)[0] + base_fname
