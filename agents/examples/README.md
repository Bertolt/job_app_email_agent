# Job Email Agent Examples

This directory contains examples demonstrating how to use the Job Email agent.

## Installation

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai/download)

### 2. Run the required LLM model

Pull if not done yet

```bash
ollama pull llama3.2
```

Run Ollama

```bash
ollama run llama3.2
```

### 3. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage Examples

### Basic Usage

Run the agent with default settings:

```bash
python agent.py
```

This will:

- Read from `messages.csv` in the current directory
- Output to `job_applications_reviewed.csv`
- Use the `llama3.2` model
- Remove duplicates based on company name and job role

### Custom Input/Output Files

Custom Input File
This uses a different CSV file as input while maintaining other defaults.

```bash
python agent.py --input_file data/messages_test_sample.csv
```

Custom Output File
This saves the results to a custom filename instead of the default.

```bash
python agent.py --output_file custom_applications.csv
```

### Using a Different Model

This uses the Mistral model instead of llama3.2 (you must have pulled this model with ollama pull mistral first).

```bash
python agent.py --model_name mistral
```

### Managing Duplicates

This allows duplicate job applications if they're more than 14 days apart.

```bash
python agent.py --allow_duplicates yes --duplicates_window 14
```

### Complete Example with All Options

This example:

```bash
python agent.py --input_file data/messages_test_sample.csv --output_file custom_output.csv --model_name mistral --allow_duplicates yes --duplicates_window 30
```

Uses messages_test_sample.csv as input
Saves results to custom_output.csv
Uses the Mistral model
Allows duplicates if they're more than 30 days apart
Processing Multiple Files

### To process several files and combine the results

```bash
python agent.py --input_file data/messages.csv --output_file results1.csv
python agent.py --input_file data/messages_test_sample.csv --output_file results2.csv
```
