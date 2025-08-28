"""Setup script for Neural SDE Pricing and Hedging package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neural-sde-pricing",
    version="1.0.0",
    author="Patryk Gierjatowicz, Marc Sabate-Vidales, David Šiška, Lukasz Szpruch, Žan Žurič",
    author_email="",
    description="Neural SDEs for robust option pricing and hedging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/NeuralSDE_pricing_hedging",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=21.0.0",
            "flake8>=3.9.0",
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neural-sde-train=nsde_LV:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.pt", "images/*.png", "images/*.pdf"],
    },
    keywords="neural-sde, option-pricing, stochastic-differential-equations, quantitative-finance, hedging",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/NeuralSDE_pricing_hedging/issues",
        "Source": "https://github.com/yourusername/NeuralSDE_pricing_hedging",
        "Documentation": "https://github.com/yourusername/NeuralSDE_pricing_hedging#readme",
        "Paper": "https://arxiv.org/abs/2007.04154",
    },
)
