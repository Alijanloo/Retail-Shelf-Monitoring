from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="retail_shelf_monitoring",
    version="0.1.0",
    author="Ali Janloo",
    author_email="mahmoodjanlooali@gmail.com",
    description=(
        "AI-Powered Retail Shelf Monitoring System for SKU Detection & "
        "Planogram Compliance"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Alijanloo/Retail-Shelf-Monitoring",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
)
