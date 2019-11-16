import setuptools


description = (
    "Covariance Matrix Adaptation Evolution Strategy (CMA-ES) implemented with TensorFlow v2"
)

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='cma-es',
    packages=['cma'],
    version='1.3.0',
    license='MIT',
    author="Romain Strock",
    author_email="romain.strock@gmail.com",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/srom/cma-es',
    keywords=['optimization', 'numerical-optimization', 'tensorflow'],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'tensorflow>=2.0',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
