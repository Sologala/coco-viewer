from setuptools import setup, find_packages

setup(
    name='cocoviewer',
    version='0.0.1',
    install_requires=[
        'numpy',
        'imagesize',
        'importlib-metadata; python_version<"3.10"',
        'pycocotools',
        "sahi"  # On Windows, Shapely needs to be installed via Conda: conda install -c conda-forge shapely
    ],
    packages=find_packages(
    ),
    entry_points={
        'console_scripts': [
            'cocoviewer = cocoviewer.cocoviewer:main',
            'cocostatic = cocoviewer.cocoutils:plotstatiscEntry',
            'cocomerge = cocoviewer.cocoutils:mergeEntry',
            'cocofix = cocoviewer.cocoutils:fix_coco_datasetEntry',
            'cocoempty = cocoviewer.cocoutils:cocoemtpyEntry',
            'cocosplit = cocoviewer.cocoutils:splitEntry',
            'cocofiltercat = cocoviewer.cocoutils:extractCatEntry',
            'cocofromyolo = cocoviewer.coco_cvt_from_yolo:yolo2coco',
            'coco_cvt_res = cocoviewer.cocoutils:results2AnnEntry',
            'coco_sample = cocoviewer.cocoutils:cocoSmapleEntry',
        ]
    }
)
