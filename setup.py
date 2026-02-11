from setuptools import setup, find_packages

setup(
    name="agent-orchestra",
    version="0.1.0",
    description="A multi-agent orchestration framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Agent Orchestra Team",
    author_email="team@agent-orchestra.dev",
    url="https://github.com/andreycpu/agent-orchestra",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "asyncio",
        "pydantic>=2.0",
        "aioredis>=2.0",
        "httpx>=0.24",
        "uvloop>=0.17",
        "structlog>=23.0",
        "prometheus-client>=0.16",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "black>=23.0",
            "flake8>=5.0",
            "mypy>=1.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)