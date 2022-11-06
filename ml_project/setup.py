from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Heart Disease Cleveland UCI classification task ',
    author='Batyrkhan Gainitdinov',
    license='MIT',
    entrypoints={
    'console_scripts': [
      'foo=src.models:main',
    ]},
)
