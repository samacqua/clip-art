import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="clip_art",
    py_modules=["clip_art"],
    version="0.0",
    description="Package for creating art with CLIP and combining different methods",
    author="Sam Acquaviva",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)