from setuptools import setup, find_packages

setup(
    name='audioForma',
    version='1.0',
    long_description='A Microservice for translating .mp3 files to data',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=['Flask','librosa']
)