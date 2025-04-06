from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="anomaly_detection_sp500",
    version="0.1.0",
    author="Fabian Beerli",
    author_email="your.email@example.com",
    description="Unsupervised Anomaly Detection in S&P 500: A Comparative Approach",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/anomaly_detection_sp500",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)