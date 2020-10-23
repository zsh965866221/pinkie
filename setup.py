from setuptools import setup, find_packages

reqiured_packages = [
  'numpy'
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
