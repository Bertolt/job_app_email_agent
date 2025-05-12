"""
This module contains tests for the email job agent functionality.
"""

import json
import os
import tempfile
import pandas as pd

import pytest

from agents.job_email_agent import JobApplicationEmailAgent


# Fixtures
@pytest.fixture
def sample_text():
    return """Dear John Doe,

    Thank you for your application for the Software Engineer position at XYZ Corp.
    We received your application on 15-04-2025.

    Best regards,
    HR Team
    XYZ Corp
    """


@pytest.fixture
def temp_files():
    """Create temporary input and output files."""
    temp_input = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".csv")
    temp_output = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".csv")

    yield temp_input.name, temp_output.name

    # Cleanup after test
    for temp_file in [temp_input.name, temp_output.name]:
        try:
            os.unlink(temp_file)
        except OSError:
            pass


@pytest.fixture
def mock_ollama_chat(monkeypatch):
    """Mock the ollama.chat function."""

    def mock_chat(*_args, **_kwargs):
        return {"message": {"content": "Hello!"}}

    monkeypatch.setattr("agents.job_email_agent.ollama.chat", mock_chat)
    return mock_chat


@pytest.fixture
def job_email_agent(temp_files, mock_ollama_chat, sample_text):
    """Create a Agent instance with temp files."""
    input_file, output_file = temp_files

    # Write sample text to temp input file
    with open(input_file, "w") as f:
        f.write(sample_text)

    return JobApplicationEmailAgent(
        input_file=input_file,
        output_file=output_file,
        model_name="test_model",
        allow_duplicates="no",
    )


def test_initialization(mock_ollama_chat):
    """Test Agent initialization."""
    job_email_agent = JobApplicationEmailAgent(
        input_file="test_input.csv",
        output_file="test_output.csv",
        model_name="test_model",
        allow_duplicates="yes",
        duplicates_window=14,
    )

    assert job_email_agent.input_file == "test_input.csv"  # nosec B101
    assert job_email_agent.output_file == "test_output.csv"  # nosec B101
    assert job_email_agent.model_name == "test_model"  # nosec B101
    assert job_email_agent.allow_duplicates == "yes"  # nosec B101
    assert job_email_agent.duplicates_window == 14  # nosec B101


def test_validate_ollama_connection_success(job_email_agent, monkeypatch):
    """Test successful Ollama connection."""
    # Define a mock that tracks calls
    calls = []

    def mock_chat(*args, **kwargs):
        calls.append((args, kwargs))
        return {"message": {"content": "Hello!"}}

    monkeypatch.setattr("agents.job_email_agent.ollama.chat", mock_chat)

    # This should not raise any exception
    job_email_agent.validate_ollama_connection()

    # Check that the function was called with expected parameters
    assert len(calls) == 1  # nosec B101
    _, kwargs = calls[0]
    assert kwargs["model"] == "test_model"  # nosec B101
    assert kwargs["messages"] == [{"role": "user", "content": "Hello!"}]  # nosec B101


def test_validate_ollama_connection_failure(job_email_agent, monkeypatch):
    """Test failed Ollama connection."""

    def mock_chat(*args, **kwargs):
        raise Exception("Connection failed")

    monkeypatch.setattr("agents.job_email_agent.ollama.chat", mock_chat)

    with pytest.raises(SystemExit):
        job_email_agent.validate_ollama_connection()


def test_extract_info(job_email_agent, monkeypatch, sample_text):
    """Test extract_info method."""
    expected_json = {
        "date": "15-04-2025",
        "company_name": "XYZ Corp",
        "job_role": "Software Engineer",
    }

    def mock_chat(*args, **kwargs):
        return {"message": {"content": json.dumps(expected_json)}}

    monkeypatch.setattr("agents.job_email_agent.ollama.chat", mock_chat)

    result = job_email_agent.extract_info(sample_text)
    assert result == json.dumps(expected_json)  # nosec B101


def test_process_messages(job_email_agent, monkeypatch):
    """Test process_messages method."""
    expected_json = {
        "date": "15-04-2025",
        "company_name": "XYZ Corp",
        "job_role": "Software Engineer",
    }

    def mock_extract_info(self, text):
        return json.dumps(expected_json)

    monkeypatch.setattr(JobApplicationEmailAgent, "extract_info", mock_extract_info)

    # Clear the DataFrame before processing
    job_email_agent.reviewed_df = pd.DataFrame(
        columns=["date", "company_name", "job_role"]
    )

    # Write the sample text in CSV format WITHOUT a header row
    with open(job_email_agent.input_file, "w", encoding="utf-8") as f:
        # Skip the header row
        f.write(
            '"Test Subject","test@example.com","user@example.com",'
            '"15-04-2025","*","12345","This is a test email body"'
        )

    # Process messages
    job_email_agent.process_messages()

    # Check that the dataframe has the expected values
    assert len(job_email_agent.reviewed_df) == 1  # nosec B101
    assert (
        job_email_agent.reviewed_df.iloc[0]["company_name"] == "XYZ Corp"
    )  # nosec B101
    assert (
        job_email_agent.reviewed_df.iloc[0]["job_role"] == "Software Engineer"
    )  # nosec B101


def test_eliminate_duplicates_no_duplicates(job_email_agent):
    """Test eliminate_duplicates with allow_duplicates=no."""
    # Create dataframe with duplicates
    job_email_agent.reviewed_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["15-04-2025", "16-04-2025", "17-04-2025"]),
            "company_name": ["XYZ Corp", "XYZ Corp", "ABC Inc"],
            "job_role": ["Software Engineer", "Software Engineer", "Data Scientist"],
        }
    )

    job_email_agent.allow_duplicates = "no"
    job_email_agent.eliminate_duplicates()

    # Should have dropped one duplicate
    assert len(job_email_agent.reviewed_df) == 2  # nosec B101


def test_eliminate_duplicates_with_time_window(job_email_agent):
    """Test eliminate_duplicates with allow_duplicates=yes and time window."""
    # Create dataframe with duplicates at different dates
    job_email_agent.reviewed_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-15", "2025-04-20", "2025-04-30"]),
            "company_name": ["XYZ Corp", "XYZ Corp", "XYZ Corp"],
            "job_role": ["Software Engineer", "Software Engineer", "Software Engineer"],
        }
    )

    job_email_agent.allow_duplicates = "yes"
    job_email_agent.duplicates_window = 7
    job_email_agent.eliminate_duplicates()

    # Should keep entries more than 7 days apart
    assert len(job_email_agent.reviewed_df) == 2  # nosec B101


def test_write_to_csv(job_email_agent, monkeypatch):
    """Test write_to_csv method."""
    # Create a simple dataframe
    job_email_agent.reviewed_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-15"]),
            "company_name": ["XYZ Corp"],
            "job_role": ["Software Engineer"],
        }
    )

    # Track calls to to_csv
    called_with = {}

    def mock_to_csv(self, *args, **kwargs):
        called_with.update(kwargs)
        called_with["file"] = args[0]

    monkeypatch.setattr("pandas.DataFrame.to_csv", mock_to_csv)

    job_email_agent.write_to_csv()

    # Check that to_csv was called with the right parameters
    assert called_with["file"] == job_email_agent.output_file  # nosec B101
    assert called_with["index"] is False  # nosec B101
    assert called_with["mode"] == "w"  # nosec B101
    assert called_with["header"] is True  # nosec B101


def test_json_decode_error_handling(job_email_agent, monkeypatch):
    """Test handling of JSON decode errors."""

    # Return invalid JSON
    def mock_chat(*_args, **_kwargs):
        return {"message": {"content": "Not valid JSON"}}

    monkeypatch.setattr("agents.job_email_agent.ollama.chat", mock_chat)

    # Make sure there's content in the file to process
    with open(job_email_agent.input_file, "w", encoding="utf-8") as f:
        f.write(
            '"Test Subject","test@example.com","user@example.com",'
            '"15-04-2025","*","12345","This is a test email body"'
        )

    # Track calls to to_csv for error logging
    called = []

    def mock_error_csv(*_args, **_kwargs):
        called.append(True)

    monkeypatch.setattr("pandas.DataFrame.to_csv", mock_error_csv)

    # This should catch and log the JSON decode error
    job_email_agent.process_messages()

    # Check that error was logged to CSV
    if len(called) <= 0:
        pytest.fail("Expected at least one call to error logging CSV")
