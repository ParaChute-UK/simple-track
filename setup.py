from setuptools import setup

setup(
    name='simple_track',
    description='Simple cloud tracking',
    license='LICENSE',
    packages=[
        'simple_track',
    ],
    install_requires=[
        'matplotlib',
        'networkx',
        'numpy',
    ]
)
