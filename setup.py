from setuptools import setup, find_packages

setup(
    name='cocoviewer',
    version='0.0.1',
    install_requires=[
        'numpy',
        'importlib-metadata; python_version<"3.10"',
    ],
    packages=find_packages(
        where="pyp",
        include=["*"]
    ),
    entry_points={
        'console_scripts': [
            'cocoviewer = cocoviewer.cocoviewer:main',
        ]
    }
)
