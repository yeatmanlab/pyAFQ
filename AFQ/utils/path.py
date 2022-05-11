def drop_extension(path):
    base_fname = path.split('/')[-1].split('.')[0]
    path.split(base_fname)[0] + base_fname
    return path
