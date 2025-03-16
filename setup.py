import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cdd",
    version="0.1.0",
    author="Hygo",
    author_email="hygo2025@gmail.com",
    description="CDD - Comparação de sistemas de recomendação",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hygo2025/cdd-projeto-final",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        # Dependências principais
        "category-encoders>=2.6.0,<3",
        "cornac>=2.3.0,<3; python_version>='3.9'",
        "hyperopt>=0.2.7,<1",
        "lightgbm>=4.0.0,<5",
        "locust>=2.12.2,<3",
        "memory-profiler>=0.61.0,<1",
        "nltk>=3.8.1,<4",
        "numba>=0.57.0,<1",
        "pandas>2.0.0,<3.0.0",
        "pandera[strategies]>=0.15.0; python_version>='3.9'",
        "retrying>=1.3.4,<2",
        "scikit-learn>=1.2.0,<2",
        "scikit-surprise>=1.1.3",
        "seaborn>=0.13.0,<1",
        "statsmodels>=0.14.4; python_version>='3.9'",
        "transformers>=4.27.0,<5",
        # Outras dependências do projeto
        "dynaconf",
        "tqdm",
        "requests",
        "joblib"
    ],
    extras_require={
        "gpu": [
            "fastai==2.7.19",
            "nvidia-ml-py>=11.525.84",
            "spacy<=3.7.5; python_version<='3.8'",
            "tensorflow>=2.8.4,!=2.9.0.*,!=2.9.1,!=2.9.2,!=2.10.0.*,<2.16",
            "tf-slim>=1.1.0",
            "torch>=2.0.1,<3",
            "torchvision",
            "torchaudio",
            "recommenders[gpu]",
            "tensorflow[and-cuda]",
            "tensorrt"
        ],
        "spark": [
            "pyarrow>=10.0.1",
            "pyspark>=3.3.0,<4.0.0"
        ]
    }
)
