from setuptools import setup


setup(
    name="facial-recognition-system",
    version="1.0.0",
    description="An adapted face detection and recognition project.",
    packages=["src"],
    install_requires=[
        "numpy<2.3",
        "opencv-python>=4.8",
        "dlib>=19.24",
        "face-recognition>=1.3.0",
        "scikit-learn>=1.3",
        "joblib>=1.3",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.13",
        "setuptools<81",
    ],
)
