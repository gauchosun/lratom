import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def find_packages_with_dir(src_base, exclude):
    """Find packages under the given base directory and append their paths.
    """
    pkgs = setuptools.find_packages(src_base, exclude)
    return {pkg:src_base + '/' + pkg.replace('.', '/') for pkg in pkgs}


pkgs_dir = find_packages_with_dir('src/python', exclude=[])


setuptools.setup(
    name="lratom",
    version="0.1.1",
    author="Atom Sun",
    author_email="nsatom@163.com",
    description="Linear Regression Atomic.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gauchosun/lratom.git",
    packages=pkgs_dir.keys(),
    package_dir=pkgs_dir,
)
