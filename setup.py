import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
        name = "XNet",
        version = "0.0.1",
        author = "Joseph Bullock, Carolina Cuesta-Lazaro, & Arnau Quera-Bofarull",
        author_email = "j.p.bullock@durham.ac.uk",
        description = "A CNN to segment X-Ray images",
        long_description = long_description,
        long_description_content_type = "text/markdown",
        url = "https://github.com/josephpb/xnet",
        packages = setuptools.find_packages(),
        classifiers = [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            ],
        install_requires =  setuptools.find_packages(exclude=('tests', 'docs'))
        )

