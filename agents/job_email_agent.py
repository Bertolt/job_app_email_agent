"""
Agent for automated job application email processing.

This agent autonomously processes CSV files containing job application emails,
leverages natural language models to extract essential information (date, company, role),
manages duplicate entries based on custom rules, and maintains a structured CSV record
of processed applications.

Parameters:
    input_file (str): Path to the input CSV file containing email data.
    output_file (str): Path to save the processed job application data.
    model_name (str): Name of the Ollama model to use for information extraction.
    allow_duplicates (str): Whether to allow duplicate job applications ('yes' or 'no').
    duplicates_window (int, optional): If allowing duplicates, the number of days
"""

import pandas as pd
import ollama
import sys
import os
import json
import argparse
import shutil

import re
import unicodedata

# from rapidfuzz import fuzz


from agents.agent import Agent

PUNCTUATION = re.compile(r"[^\w\s-]+", re.UNICODE)
MULTI_WHITESPACE = re.compile(r"\s+")
CROP_TAIL_AFTER_DASH = re.compile(r"\s+-\s+.*$")

# any mixture of “m.b.H.”, “mbh”, “MBH” …, plus the usual suspects
LEGAL_SUFFIXES = re.compile(
    r"""\b(
           gmbh        | m\s*\.?\s*b\s*\.?\s*h\s*\.? | mbh |
           ag | se | kg | ug |
           llc | inc | corp\.? | company | co\.? |
           ltd | plc | oy | sas | sa | sarl | pte\.?
       )\b""",
    re.I | re.X,
)

GENDER_MARKERS = re.compile(r"\((?:m/w/d|w/m/d|m/f/d|d/f/m)\)", re.I)


class JobApplication:
    """
    Class representing a job application with attributes for date, company name, and job role.
    """

    def __init__(self, date, company_name, job_role):
        self.date = date
        self.company_name = company_name
        self.job_role = job_role
        # status are sent > active > rejected > accepted
        self.status = None
        self.rejection_reason = None

    def __repr__(self):
        return (
            f"JobApplication(date={self.date}, "
            f"company_name={self.company_name}, "
            f"job_role={self.job_role})"
        )


class JobApplicationEmailAgent(Agent):
    """
    Agent for extracting structured data from job application emails.

    This class processes emails from CSV files containing job application confirmations,
    extracts key information (date, company name, job role) using a language model,
    eliminates duplicates based on configurable criteria, and saves the processed data
    to a structured CSV file.

    Parameters:
        input_file (str): Path to the input CSV file containing email data.
        output_file (str): Path to save the processed job application data.
        model_name (str): Name of the Ollama model to use for information extraction.
        allow_duplicates (str): Whether to allow duplicate job applications ('yes' or 'no').
        duplicates_window (int, optional): If allowing duplicates, the number of days
    """

    name = "Job Application Email Agent"
    color = Agent.GREEN

    def __init__(
        self,
        input_file="messages.csv",
        output_file="job_applications_reviewed.csv",
        model_name="llama3.2",
        allow_duplicates="no",
        duplicates_window=None,
    ):
        super().__init__()

        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        self.allow_duplicates = allow_duplicates
        self.duplicates_window = duplicates_window

        self.log.info("Job Application Email Agent initialized")
        # If file existing progress move file to backup timestamp + namefile
        if os.path.exists(self.output_file):
            try:
                # First create a backup of the existing file
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                file_name, file_ext = os.path.splitext(self.output_file)
                backup_file = f"{file_name}_{timestamp}{file_ext}"
                # Copy the file to backup
                shutil.copy2(self.output_file, backup_file)
                self.log.info(f"Created backup of existing output file: {backup_file}")
            except Exception as e:
                self.log.error(f"Failed to create backup of existing output file: {e}")
                sys.exit(1)

        self.reviewed_df = pd.DataFrame(
            columns=["date", "company_name", "job_role", "status", "rejection_reason"]
        )
        self.reviewed_ids = set()

        # Test connection to the Ollama model
        self.validate_ollama_connection()

    def validate_ollama_connection(self):
        """
        Validates the connection to the Ollama model by attempting a simple chat interaction.

        This method tests if the specified Ollama model is accessible and functioning by sending
        a basic "Hello!" message. If the connection fails, the program will log an error and exit.
        """
        try:
            ollama.chat(
                model=self.model_name, messages=[{"role": "user", "content": "Hello!"}]
            )
            self.log.info(
                "Successfully connected to the Ollama model: %s", self.model_name
            )
        except Exception as e:
            self.log.error("Failed to connect to the Ollama model: %s", self.model_name)
            self.log.error("Error: %s", e)
            exit(1)

    def extract_info(self, text):
        system_prompt = """Extract the following from a his job application email:
        - date (format: DD-MM-YYYY)
        - company name (real company, not platform)
        - job role (title of position applied for)
        - Status (sent, active, rejected, accepted)
        - rejection reason (if applicable)

        For job role:
        - Exclude any gender in the such as (d/f/m) or (w/m/d).
        - Exclude information regarding the location of the job.

        IMPORTANT: Your response must be valid JSON with no additional text.
        Generate the output as JSON do not include any other text.
        The JSON should be in the following format:
        {
            "date": "DD-MM-YYYY",
            "company_name": "Company Name",
            "job_role": "Job Role"
            "status": "sent/active/rejected/accepted",
            "rejection_reason": "Reason for rejection (if applicable)"
        }
        """
        # User prompt with the actual email content
        user_prompt = f"Extract information from this job application email:\n\n{text}"

        # Call Ollama with both system and user prompts
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"]

    def process_messages(self):
        """Process messages from a CSV file with multiple emails, each potentially
        containing CRLF characters."""
        try:
            # Use pandas to read the CSV file with proper handling of quoted fields that contain
            # newlines
            df = pd.read_csv(
                self.input_file,
                header=None,
                quotechar='"',
                engine="python",
                on_bad_lines="skip",
            )

            # Check if the file is successfully read as a DataFrame
            if df.empty:
                self.log.warning("No data found in file: %s", self.input_file)
                return

            self.log.info("Successfully read file with %d rows", len(df))

            # Process each row as a separate email
            for idx, row in df.iterrows():
                # Extract the email content and add debug output
                email_content = row.iloc[-1] if len(row) > 0 else None

                if isinstance(email_content, str) and email_content.strip():
                    try:
                        # Extract information from the email content
                        suggestion = self.extract_info(email_content)
                        self.log.info("Processed row %d: %s", idx, suggestion)

                        try:
                            # Parse the suggestion as JSON
                            suggestion_dict = json.loads(suggestion)
                            # Check for duplicates
                            company_name = self.normalise_company(
                                suggestion_dict.get("company_name", None)
                            )
                            job_role = self.normalise_role(
                                suggestion_dict.get("job_role", None)
                            )
                            # Add it to the dataframe

                            self.reviewed_df.loc[len(self.reviewed_df)] = [
                                pd.to_datetime(
                                    suggestion_dict.get("date", None), format="%d-%m-%Y"
                                ),
                                company_name,
                                job_role,
                                suggestion_dict.get("status", None),
                                suggestion_dict.get("rejection_reason", None),
                            ]
                        except json.JSONDecodeError as e:
                            self.log.error(
                                "Failed to parse JSON from suggestion for row %d: %s",
                                idx,
                                e,
                            )
                            # Log the error
                            self._log_error(e, email_content, suggestion, idx)

                    except Exception as e:
                        self.log.error("Failed to process row %d: %s", idx, e)
                        self._log_error(e, email_content, None, idx)

        except Exception as e:
            self.log.error("Error reading file: %s", e)

    def normalise_company(self, company_name: str) -> str:
        company_name = (
            unicodedata.normalize("NFKD", company_name)
            .encode("ascii", "ignore")
            .decode()
        )
        company_name = company_name.lower()

        # kill “- Munich, BY” or similar location hints
        company_name = CROP_TAIL_AFTER_DASH.sub("", company_name)

        # remove punctuation *before* looking for suffixes with dots
        company_name = PUNCTUATION.sub(" ", company_name)
        company_name = LEGAL_SUFFIXES.sub("", company_name)

        company_name = MULTI_WHITESPACE.sub(" ", company_name).strip("- ").strip()

        # optional: keep only the “brand” (first token or first two short tokens)
        words = company_name.split()
        if len(words) > 2:
            company_name = words[0] if "-" in words[0] else " ".join(words[:2])

        return company_name

    def normalise_role(self, role: str) -> str:
        txt = GENDER_MARKERS.sub("", role)
        txt = PUNCTUATION.sub(" ", txt)
        txt = MULTI_WHITESPACE.sub(" ", txt).strip()
        return txt.lower()

    def _log_error(self, error, content_excerpt, suggestion=None, row_idx=None):
        """Log processing errors to a CSV file."""
        data = {
            "error": [str(error)],
            "row_idx": [row_idx],
            "content_excerpt": [content_excerpt],
        }

        if suggestion:
            data["suggestion"] = [suggestion]

        error_df = pd.DataFrame(data)
        error_df.to_csv("error_log.csv", mode="a", header=False, index=False)

    def eliminate_duplicates(self):
        """Eliminate duplicates from the DataFrame using a hash map."""
        if self.allow_duplicates == "no":
            # Drop duplicates based on company_name and job_role
            self.reviewed_df.drop_duplicates(
                subset=["company_name", "job_role"], inplace=True, keep="first"
            )
        elif self.allow_duplicates == "yes":
            if self.duplicates_window is not None:
                # Ensure the date column is in datetime format
                self.reviewed_df["date"] = pd.to_datetime(self.reviewed_df["date"])
                self.reviewed_df.sort_values(by="date", inplace=True)

                # Use a hash map to track the most recent entry for each (company_name, job_role)
                company_job_date_tracker = {}
                rows_to_drop = []

                for idx, row in self.reviewed_df.iterrows():
                    key = (row["company_name"], row["job_role"])
                    if key in company_job_date_tracker:
                        # Check if the date difference is within the duplicates_window
                        last_date = company_job_date_tracker[key]
                        if (row["date"] - last_date).days <= self.duplicates_window:
                            rows_to_drop.append(idx)
                        else:
                            company_job_date_tracker[key] = row[
                                "date"
                            ]  # Update the most recent date
                    else:
                        company_job_date_tracker[key] = row[
                            "date"
                        ]  # Add new entry to the hash map

                # Drop the rows marked as duplicates
                self.reviewed_df.drop(index=rows_to_drop, inplace=True)
            else:
                # If no custom time frame is provided, drop duplicates based on all rows
                self.reviewed_df.drop_duplicates(
                    subset=["company_name", "job_role"], inplace=True, keep="first"
                )

    def write_to_csv(self):
        """Write the DataFrame to a CSV file."""
        try:
            if self.allow_duplicates == "yes":
                self.eliminate_duplicates()
            # Save the DataFrame to a CSV file
            self.reviewed_df.to_csv(
                self.output_file, index=False, mode="w", header=True
            )
            self.log.info("Data saved to %s", self.output_file)
        except Exception as e:
            self.log.error("Failed to save data to %s: %s", self.output_file, e)


# Example usage with command-line arguments
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Process job application emails.")
    arg_parser.add_argument(
        "--input_file",
        type=str,
        default="messages.csv",
        help="Path to the input file (default: messages.csv)",
    )
    arg_parser.add_argument(
        "--output_file",
        type=str,
        default="job_applications_reviewed.csv",
        help="Path to the output file (default: job_applications_reviewed.csv)",
    )

    arg_parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.2",
        help="Name of the Ollama model to use (default: llama3.2)",
    )
    arg_parser.add_argument(
        "--allow_duplicates",
        type=str,
        default="no",
        choices=["yes", "no"],
        help="Time frame for eliminating duplicates (default: all)",
    )
    arg_parser.add_argument(
        "--duplicates_window",
        type=int,
        default=None,
        help="Number of days for custom time frame (required if allow_duplicates is 'active')",
    )

    args = arg_parser.parse_args()

    email_agent = JobApplicationEmailAgent(
        input_file=args.input_file, output_file=args.output_file
    )

    email_agent.process_messages()
    email_agent.write_to_csv()
