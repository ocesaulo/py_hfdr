import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='py_hfdr',
    version='0.0.1',
    description='Tools to handle high frequency doppler radar data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/ocesaulo/py_hfdr',
    author='Saulo M Soares',
    author_email='ocesaulo@gmail.com',
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)

