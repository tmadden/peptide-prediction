from setuptools import setup

setup(
    name="peptide",
    setup_requires=["pytest-runner", "setuptools-git-version"],
    install_requires=["numpy"],
    tests_require=["pytest", "codecov", "scikit-learn>=0.20"],
    extras_require={"sklearn": ["scikit-learn>=0.20"]},
    packages=["peptide"],
    package_dir={"": "src"},
    version="0",
    # version_format="{tag}.dev{commitcount}+{gitsha}",
)
