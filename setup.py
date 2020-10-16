from setuptools import setup, find_packages

reqiured_packages = [
  'numpy',
  'torch>=1.6.0'
]

setup(
  name='pinkie',
  version='0.0.1',
  author='pinkie',
  packages=find_packages(),
  install_requires=reqiured_packages,
  include_package_data=True,
  zip_safe=False
)
