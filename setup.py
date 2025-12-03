from setuptools import setup, find_packages

setup(
    name="sae-music-interp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "audiocraft>=1.0.0",
        "librosa>=0.10.0",
        "scikit-learn>=1.3.0",
        "gradio>=3.35.0",
        "tqdm>=4.65.0",
    ],
    author="Siyu Zhang",
    author_email="zhangsiyu@utexas.edu",
    description="Interpretable Music Generation via Sparse Autoencoders",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sae-music-interp",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)
