from setuptools import setup, find_packages

setup(
    name="videogen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'einops',
        'imageio',
        'ipython',
        'matplotlib',
        'numpy',
        'pytorch_lightning',
        'PyYAML',
        'torch',
        'torchvision',
        'tqdm',
        'wandb'
    ],
)