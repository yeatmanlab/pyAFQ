from setuptools import setup
from setuptools.command.install import install
import string
import subprocess
import os.path as op
import glob
from setuptools_scm import get_version


def local_version(version):
    """
    Patch in a version that can be uploaded to test PyPI
    """
    scm_version = get_version()
    if "dev" in scm_version:
        gh_in_int = []
        for char in version.node:
            if char.isdigit():
                gh_in_int.append(str(char))
            else:
                gh_in_int.append(str(string.ascii_letters.find(char)))
        return "".join(gh_in_int)
    else:
        return ""


class InstallpyAFQandFastSurfer(install):
    """Customized setuptools install command which updates git submodules."""

    def run(self):
        # Ensure submodules are updated and initialized
        subprocess.run(['git', 'submodule', 'update',
                       '--init', '--recursive'], check=True)
        # Call the original install command
        install.run(self)


opts = dict(
    use_scm_version={"root": ".", "relative_to": __file__,
                     "write_to": op.join("AFQ", "version.py"),
                     "local_scheme": local_version},
    scripts=[op.join('bin', op.split(f)[-1]) for f in glob.glob('bin/*')],
    cmdclass={'install': InstallpyAFQandFastSurfer},)


if __name__ == '__main__':
    setup(**opts)
