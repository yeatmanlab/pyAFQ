from setuptools import setup


opts = dict(use_scm_version={"root": ".", "relative_to": __file__,
                             "write_to": "AFQ/version.py",
                             "local_scheme": local_version})


if __name__ == '__main__':
    setup(**opts)
