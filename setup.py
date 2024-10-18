from setuptools import find_packages, setup
setup(
    name='multikano',
    version='0.1.0',
    keywords=('pip', 'MultiKano', 'single-sell', 'Multi-omics'),
    description='MultiKano: an automatic cell type annotation tool for single-cell multi-omics data based on Kolmogorov–Arnold network and data augmentation',
    long_description="MultiKano provides an accurate and efficient way to automatically annotate cell typesin multi-omics datasets. All MultiKano wheels distributed on PyPI are MIT licensed.",
    license='MIT License',
    url='https://github.com/BioX-NKU/MultiKano',
    author='Siyu Li, Xinhao Zhuang, Songbo Jia',
    packages=find_packages(),
    python_requires='>3.6.0',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics'],
    install_requires=[
        'numpy>=1.22.3',
        'pandas>=1.4.3',
        'scipy>=1.9.0',
        'scikit-learn>=1.1.2',
        'numba>=0.57.1',
        'scanpy>=1.9.1',
        'anndata>=0.8.0',
        'episcanpy>=0.3.2',
        'torch>=1.11.0',

    ]
)