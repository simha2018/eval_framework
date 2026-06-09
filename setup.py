from setuptools import find_packages, setup

setup(
    name="genai_eval",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "anthropic>=0.40.0",
        "openai>=1.50.0",
        "pydantic>=2.0.0",
    ],
    python_requires=">=3.10",
)
