from setuptools import setup, find_packages

setup(
    name="agents",
    version="0.1.0",
    packages=find_packages(),  # Specify where to look for packages
    install_requires=[
        "pandas",  # Example dependency
        "ollama",  # LLM interface
        "numpy",  # Required by pandas
        "python-dateutil",  # Date parsing
        "pytz",  # Timezone handling
    ],
    entry_points={
        "console_scripts": [
            "agents=agents.job_email_agent:main",  # entry point
        ]
    },
)
