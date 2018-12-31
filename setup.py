from setuptools import setup

setup(
    name="peptide",
    install_requires=["sklearn"],
    packages=[""],
    package_dir={"": "src"},
    py_modules=[],
    version="0",
    # version_format="{tag}.dev{commitcount}+{gitsha}",
    setup_requires=["setuptools-git-version"],
)
