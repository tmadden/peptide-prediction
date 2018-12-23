from setuptools import setup

setup(
    name='peptide',
    install_requires=['sklearn'],
    packages=['peptide'],
    package_dir={'': 'src'},
    py_modules=[],
    version_format='{tag}.dev{commitcount}+{gitsha}',
    setup_requires=['setuptools-git-version']
)
