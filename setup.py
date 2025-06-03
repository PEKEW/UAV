from setuptools import setup, find_packages

setup(
    name="battery_prediction",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0"
    ],
    entry_points={
        "console_scripts": [
            "battery-train=src.scripts.train:main",
        ],
    },
    zip_safe=False,
) 