from setuptools import setup

VERSION: dict[str, str] = {}

with open("arrafire/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(version=VERSION["VERSION"])
