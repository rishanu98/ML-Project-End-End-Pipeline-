# Consits of meta data about Project´s metadata and configuration

from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads the requirements from a file and returns them as a list.
    '''
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # Use strip() to remove leading/trailing whitespace
        requirements = [req for req in requirements if not req.startswith('-e .')]  # Exclude '-e .'
    return requirements

setup(
    name='ML_PROJECT',  # Avoid spaces in package names
    version='0.0.1',
    author='Anurag Shukla',
    author_email='rishnau68@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)