from pathlib import Path

from setuptools import setup

VERSION: dict[str, str] = {}


with (Path().absolute() / "arrayfire" / "version.py").open("r") as version_file:
    exec(version_file.read(), VERSION)

setup(version=VERSION["VERSION"])
