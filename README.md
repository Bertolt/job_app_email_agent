# Job Application Email Agent

As the job market becomes more competitive and requires tracking multiple applications daily, I decided to create a simple tool to manage this process.

So in a nutshel I use a small LLM application to do the job application parsing for me.

It uploads a .csv file containing email communication and outputs a csv file with the date and the application (company and role)

The agent should process the data accordingly and automatically update each application status

## Install Ollama

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai/download)

### 2. Pull the required LLM model

```bash
ollama pull llama3.2
```

## Install Python

### 1. Download Python

Download and install Python from [python.org](https://www.python.org/downloads/)

### 2. Verify Installation

Open a terminal and run:

```bash
python --version
```

## Python environment

### 1. Create a virtual environment

Create a virtual environment to isolate your project dependencies:

```bash
python -m venv venv
```

### 2. Activate the virtual environment

#### On Windows

```powershell
.\venv\Scripts\activate
```

#### On macOS/Linux

```bash
source venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

## Commands

For console prompting

```bash
ollama run llama3.2
```

## How to contribute

### Contributions are welcome

Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

Please make sure to update tests as appropriate and follow the existing coding style.

### Run Tests

Setup environment if needed!
Install the package in developement mode from project root directory:

```bash
pip install -e .
```

# Basic run

```bash
pytest -v
```

# With coverage report

```bash
pytest -v --cov=job_email_agent
```

# Generate HTML coverage report

```bash
pytest --cov=job_email_agent --cov-report=html
```

### Run Code Style and Linting

1. Activate Environment

    ```bash
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    ```

2. Install pre-commit Locally or run requirements.txt

    ```bash
    pip install pre-commit
    ```

3. Install the hook:

    ```bash
    pre-commit install
    ```

    This will set the `.pre-commit-config.yaml` file.

4. Optional: Run pre-commit manually:

    ```bash
    pre-commit run --all-files
    ```

### Run Autoformatters

Make sure virtualvironment is active and run Black:

```bash
black .
```
