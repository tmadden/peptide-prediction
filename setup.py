from setuptools import setup

setup(
    name="peptide-prediction",
    install_requires=["sklearn", "pytest"],
    packages=[""],
    package_dir={"": "src"},
    py_modules=[],
    version="0",
    # version_format="{tag}.dev{commitcount}+{gitsha}",
    setup_requires=["setuptools-git-version"],
)
