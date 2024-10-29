from setuptools import setup, find_packages

setup(
    name='cocoviewer',
    version='0.0.1',
    install_requires=[
        'numpy',
        'imagesize',
        'importlib-metadata; python_version<"3.10"',
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'cocoviewer = cocoviewer.cocoviewer:main',
            'cocofromyolo = cocoviewer.coco_cvt_from_yolo:yolo2coco',
        ]
    }
)
