from setuptools import setup, find_packages

setup(
    name="school-rag-mlops",
    version="1.0.0",
    author="Anna Novikova, Sofia Mikhaleva",
    description="MLOps pipeline for School RAG System",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "langchain>=0.1.0",
        "pandas>=2.0.0",
        "mlflow>=2.8.0",
        "dvc>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "run-pipeline=src.pipeline:main",
        ],
    },
)