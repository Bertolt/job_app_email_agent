"""
Integration test for JobApplicationEmailAgent.

This script demonstrates how to use the agent with real CSV data containing CRLF characters.
It processes a sample input file and compares the output with expected results.
"""

from agents.job_email_agent import JobApplicationEmailAgent

import sys
from pathlib import Path

import pandas as pd

# Add the project root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_integration_test():
    """Run an integration test using sample files."""
    # Set up paths
    data_dir = Path(__file__).parent.parent / "data/email_data"
    input_file = data_dir / "test_sample_duplicated_messages_with_crlf.csv"
    expected_output_file = data_dir / "test_sample_job_application_reviewed.csv"
    actual_output_file = data_dir / "test_output.csv"

    # delete any existing output file
    if actual_output_file.exists():
        actual_output_file.unlink()
        print(f"🗑️  Deleted existing output file: {actual_output_file}")

    # Ensure input files exist
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return False

    if not expected_output_file.exists():
        print(f"Expected output file not found: {expected_output_file}")
        return False

    print(f"Processing input file: {input_file}")

    # Initialize and run agent
    agent = JobApplicationEmailAgent(
        input_file=str(input_file),
        output_file=str(actual_output_file),
        model_name="llama3.2",
        allow_duplicates="no",
    )

    agent.process_messages()
    agent.eliminate_duplicates()
    agent.write_to_csv()

    print(f"Generated output file: {actual_output_file}")

    # Compare output with expected output
    try:
        # Read CSVs with proper date parsing
        expected_df = pd.read_csv(expected_output_file)
        actual_df = pd.read_csv(actual_output_file)

        # Normalize dates for comparison
        expected_df["date"] = pd.to_datetime(
            expected_df["date"], format="%Y-%m-%d"
        ).dt.strftime("%Y-%m-%d")
        if "date" in actual_df.columns:
            actual_df["date"] = pd.to_datetime(actual_df["date"]).dt.strftime(
                "%Y-%m-%d"
            )

        # Sort both dataframes for comparison
        expected_df = expected_df.sort_values(
            by=["date", "company_name", "job_role"]
        ).reset_index(drop=True)
        actual_df = actual_df.sort_values(
            by=["date", "company_name", "job_role"]
        ).reset_index(drop=True)

        # Compare DataFrames
        if expected_df.equals(actual_df):
            print("Test passed! Output matches expected results.")
            return True
        else:
            print("❌ Test failed! Output does not match expected results.")

            # Show differences
            print("\n--- Expected Output ---")
            print(expected_df)
            print("\n--- Actual Output ---")
            print(actual_df)

            # Find rows that are in expected but not in actual
            missing_rows = pd.merge(
                expected_df,
                actual_df,
                how="left",
                on=["date", "company_name", "job_role"],
                indicator=True,
                validate="one_to_one",
            )
            missing_rows = missing_rows[missing_rows["_merge"] == "left_only"].drop(
                "_merge", axis=1
            )

            if not missing_rows.empty:
                print("\n--- Missing Rows ---")
                print(missing_rows)

            # Find rows that are in actual but not in expected
            extra_rows = pd.merge(actual_df, expected_df, how="left", indicator=True)
            extra_rows = extra_rows[extra_rows["_merge"] == "left_only"].drop(
                "_merge", axis=1
            )

            if not extra_rows.empty:
                print("\n--- Extra Rows ---")
                print(extra_rows)

            return False

    except Exception as e:
        print(f"Error comparing files: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running JobApplicationEmailAgent integration test...")
    success = run_integration_test()

    if success:
        print("Integration test completed successfully.")
        sys.exit(0)
    else:
        print("Integration test failed.")
        sys.exit(1)
