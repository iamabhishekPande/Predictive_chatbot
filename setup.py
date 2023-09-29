from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="prediction-model",
    version="1.0.0",
    description="FastAPI-based machine learning model for conversational question answering",
    packages=find_packages(),
    install_requires=requirements,  # Use the requirements from requirements.txt
    entry_points={
        "console_scripts": [
            "run_prediction_model = main:app",  # Replace 'main' with your script/module name
        ],
    },
    extras_require={
        "extras": [
            # Define additional optional dependencies here
        ]
    }
)
