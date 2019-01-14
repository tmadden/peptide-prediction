from setuptools import setup

setup(
    name="pace",
    python_requires=">=3.6",
    setup_requires=["pytest-runner", "setuptools-git-version"],
    install_requires=["numpy"],
    tests_require=["pytest", "pytest-cov", "scikit-learn>=0.20"],
    extras_require={"sklearn": ["scikit-learn>=0.20"]},
    packages=["pace"],
    package_dir={"": "src"},
    version="0",
    # version_format="{tag}.dev{commitcount}+{gitsha}",
)
