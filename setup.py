from setuptools import setup, find_packages

setup(
    name="egoviz",
    version="0.1.0",
    author="Adesh Kadambi",
    author_email="adeshkadambi@gmail.com",
    packages=find_packages(),
    install_requires=["moviepy>=1.0.3", "opencv-python", "seaborn", "scikit-learn"],
)
