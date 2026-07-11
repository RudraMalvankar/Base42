from setuptools import setup, find_packages

setup(
    name="base42-ai-os",
    version="1.0.0",
    description="A deterministic, cost-aware Hybrid AI Operating System.",
    author="Rudra Malvankar",
    packages=find_packages(),
    install_requires=[
        "huggingface-hub",
        "llama-cpp-python",
        "pydantic",
        "httpx",
        "tenacity"
    ],
    entry_points={
        "console_scripts": [
            "base42=main:main"
        ]
    },
    python_requires=">=3.9",
)
