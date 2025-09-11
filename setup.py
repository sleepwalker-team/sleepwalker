from setuptools import setup, find_packages

setup(
    name='sleepwalker',
    version='0.1',
    description='sleepwalker',
    author='Sebastian Buschjäger and Matthias Jakobs',
    author_email='sleepwalker.lamarr.cs@tu-dortmund.de',
    url='https://github.com/sleepwalker-team/sleepwalker',
    python_requires='>=3.7',
    packages=find_packages('.')
)