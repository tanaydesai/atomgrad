import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="atomgrad",
    version="0.2.6",
    author="Tanay Desai",
    author_email="tanaydesai40@gmail.me",
    description="An autocgrad engine that is between micrograd and tinygrad with a PyTorch-like neural network API:)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tanaydesai/atomgrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)