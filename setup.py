from setuptools import setup

setup(
    name='puma',
    install_requires=[
        'appdirs', 'click', 'click_log', 'docker', 'pyyaml', 'requests',
        'websocket_client', 'msgpack'
    ],
    packages=['puma', 'cli'],
    package_dir={'': 'src'},
    py_modules=[],
    version_format='{tag}.dev{commitcount}+{gitsha}',
    setup_requires=['setuptools-git-version'],
    entry_points='''
        [console_scripts]
        puma=cli.main:cli
    ''',
)
