from setuptools import setup, find_packages

setup(
    name='transfactor',
    version='0.1.0',
    description='Transformer-based factorization models for tabular data',
    author='Tamara Cucumides',
    author_email='tacucumides@gmail.com',
    url='https://github.com/TamaraCucumides/Transfactor',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'torch>=1.10.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
