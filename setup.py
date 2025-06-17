from setuptools import setup, find_packages

setup(
    name='cutscenes',
    version='0.1.0',
    packages=find_packages(),  # auto-discovers 'models', 'training', etc.
    install_requires=[
       "torch>=2.7.0",
       "torchaudio>=2.7.0",
       "numpy>=2.2.5",
       "pillow>=11.2.1",
       "scipy>=1.15.3",
       "soundfile>=0.13.1"
    ],
    author='Daniel Gonzálbez Biosca',
    description='A project for cut-scene prediction using ML',
    python_requires='>=3.10',
)