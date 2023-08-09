from collections import defaultdict
from pathlib import Path

from setuptools import find_packages, setup

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release


def parse_requirements_file(path: Path, allowed_extras: set = None, include_all_extra: bool = True):
    requirements = []
    extras = defaultdict(list)
    with path.open("r") as requirements_file:
        import re

        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL dependencies should be handled."""
            m = re.match(r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git", req)
            if m is None:
                return req
            else:
                return f"{m.group('name')} @ {req}"

        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            req, *needed_by = line.split("# needed by:")
            req = fix_url_dependencies(req.strip())
            if needed_by:
                for extra in needed_by[0].strip().split(","):
                    extra = extra.strip()
                    if allowed_extras is not None and extra not in allowed_extras:
                        raise ValueError(f"invalid extra '{extra}' in {path}")
                    extras[extra].append(req)
                if include_all_extra and req not in extras["all"]:
                    extras["all"].append(req)
            else:
                requirements.append(req)
    return requirements, extras


ABS_PATH = Path().absolute()
# exec is used here so we don't import arrayfire whilst setting up
VERSION = {}  # type: ignore
with (ABS_PATH / "arrayfire" / "version.py").open("r") as version_file:
    exec(version_file.read(), VERSION)

# Load requirements.
install_requirements, extras = parse_requirements_file(ABS_PATH / "requirements.txt")
dev_requirements, dev_extras = parse_requirements_file(
    ABS_PATH / "dev-requirements.txt", allowed_extras={"examples"}, include_all_extra=False
)
extras["dev"] = dev_requirements
extras.update(dev_extras)

setup(
    name="arrayfire",
    version=VERSION["VERSION"],
    description="ArrayFire Python Wrapper",
    licence="BSD",
    long_description=(ABS_PATH / "README.md").open("r").read(),
    long_description_content_type="text/markdown",
    author="ArrayFire",
    author_email="technical@arrayfire.com",
    url="http://arrayfire.com",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
    ],
    keywords="arrayfire parallel computing gpu cpu opencl",
    packages=find_packages(),
    install_requires=install_requirements,
    extras_require=extras,
    include_package_data=True,
    python_requires=">=3.8.0",
    zip_safe=False,
)
