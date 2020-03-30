from setuptools import setup

setup(
    name="pace",
    python_requires=">=3.6",
    setup_requires=["pytest-runner", "setuptools-git-version"],
    install_requires=["numpy", "pandas"],
    tests_require=["pytest", "pytest-cov", "scikit-learn>=0.20"],
    extras_require={
        "sklearn": ["scikit-learn>=0.20"],
        "docs": ["sphinx"],
        "keras": ["keras>=2.2.4"]
    },
    packages=["pace"],
    package_dir={"": "src"},
    package_data={
        'pace': ['data/*.txt'],
    },
    version_format="{tag}.dev{commitcount}+{gitsha}",
)
