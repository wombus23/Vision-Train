"""
Setup script for News Image Classifier package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="news-image-classifier",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A web scraping and CNN-based image classification project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/news-image-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.15.0",
        "keras>=2.15.0",
        "beautifulsoup4>=4.12.2",
        "selenium>=4.15.2",
        "requests>=2.31.0",
        "opencv-python>=4.8.1",
        "Pillow>=10.1.0",
        "numpy>=1.24.3",
        "pandas>=2.1.3",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "PyYAML>=6.0.1",
        "tqdm>=4.66.1",
        "webdriver-manager>=4.0.1",
        "scikit-learn>=1.3.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "scrape-images=scripts.scrape_images:main",
            "train-model=scripts.train_model:main",
            "predict-image=scripts.predict:main",
        ],
    },
)