from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='sam',
    version="0.0.1",
    author='Olivia Hsu',
    author_email='oliviahsu1107@gmail.com',
    description='Sparse Abstract Machine',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/weiya711/sam",
    python_requires=">=3.5",
    packages=[
        "sam",
        "sam.sim",
    ],
    install_requires=[]
)
