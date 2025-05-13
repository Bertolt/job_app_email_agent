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
"""

import pandas as pd
import numpy as np
import ollama
import sys
import os
import json
import argparse
import shutil

import re
import unicodedata
from pydantic import BaseModel

from rapidfuzz import fuzz
from rapidfuzz.process import cdist

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


class JobApplication(BaseModel):
    """
    Class representing a job application with attributes for date, company name, and job role.
    """

    date: str
    company_name: str
    job_role: str
    status: str
    rejection_reason: str


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
    """

    name = "Job Application Email Agent"
    color = Agent.GREEN

    def __init__(
        self,
        input_file="messages.csv",
        output_file="job_applications_reviewed.csv",
        model_name="llama3.2",
        allow_duplicates="no",
    ):
        super().__init__()

        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        self.allow_duplicates = allow_duplicates

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
        system_prompt = """Extract the following from this job application email:
            - date (format: DD-MM-YYYY)
            - company name (real company, not platform)
            - job role (title of position applied for)
            - Status (sent, active, rejected, accepted)
            - rejection reason (if applicable)

            For job role:
            - Exclude any gender in the such as (d/f/m) or (w/m/d).
            - Exclude information regarding the location of the job.

            IMPORTANT: Your response must be ONLY valid JSON with no additional text.
            Do not include code fences, markdown formatting, or any explanatory text.
            The JSON should be in the following format:
            {
                "date": "DD-MM-YYYY",
                "company_name": "Company Name",
                "job_role": "Job Role",
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
            format=JobApplication.model_json_schema(),  # JSON guard-rail
            options={"temperature": 0.0},
        )

        suggestion = JobApplication.model_validate_json(response["message"]["content"])
        # Check if the response is valid JSON
        return suggestion

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
                    # Extract information from the email content
                    suggestion = self.extract_info(email_content)
                    self.log.info("Processed row %d: %s", idx, suggestion)

                    try:
                        company_name = self.normalise_company(suggestion.company_name)
                        job_role = self.normalise_role(suggestion.job_role)
                        # Add it to the dataframe
                        date_str = self.parse_date(suggestion.date)
                        self.reviewed_df.loc[len(self.reviewed_df)] = [
                            date_str,
                            company_name,
                            job_role,
                            suggestion.status,
                            suggestion.rejection_reason,
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
            self.log.error("Error reading file: %s", e)

    def parse_date(self, date_str: str) -> str:
        if date_str:
            date_str = date_str.replace("/", "-")
            try:
                parsed_date = pd.to_datetime(date_str, format="%Y-%m-%d")
            except ValueError:
                # Fallback to more flexible parsing
                parsed_date = pd.to_datetime(date_str, dayfirst=True)
            return parsed_date.normalize().strftime("%Y-%m-%d")
        else:
            return ""

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
        """Second step on data analysis.
        when the dataframe is already created, we can eliminate duplicates
        """
        if self.reviewed_df.empty:
            self.log.info("No data to check for duplicates")
            return

        original_count = len(self.reviewed_df)
        self.log.info(f"Checking for duplicates in {original_count} entries")

        # Create normalized keys for fuzzy matching
        df = self.reviewed_df.copy()

        # First remove exact duplicates
        df.drop_duplicates(subset=["company_name", "job_role", "status"], inplace=True)

        keys = (
            df[["company_name", "job_role"]]
            .astype(str)
            .agg(" ".join, axis=1)
            .to_numpy()
        )

        # Create a distance matrix for fuzzy matching
        sim_matrix = (
            cdist(
                keys,
                keys,
                scorer=fuzz.token_set_ratio,
                workers=-1,  # melt the CPU: use all cores
                score_cutoff=95,
            )  # returns 0 for “not similar”
            > 0  # convert scores to True/False
        )

        # Exact match on status  (broadcast to a matrix)
        status_eq = df["status"].to_numpy()
        status_matrix = status_eq[:, None] == status_eq[None, :]

        # A duplicate exists when *both* conditions are true
        dupe_matrix = sim_matrix & status_matrix
        # Mark rows that match any other row (ignore the diagonal)
        np.fill_diagonal(dupe_matrix, False)
        mask_dupes = dupe_matrix.any(axis=1)

        # Keep only the unique rows
        df = df.loc[~mask_dupes].reset_index(drop=True)

        # write to CSV using test_output_no_dupes.csv
        df.to_csv(
            "test_output_no_dupes.csv",
            index=True,
            mode="w",
            header=True,
        )

    def write_to_csv(self):
        """Write the DataFrame to a CSV file."""
        try:
            if self.allow_duplicates == "yes":
                self.eliminate_duplicates()
            # Save the DataFrame to a CSV file
            self.reviewed_df.to_csv(self.output_file, index=True, mode="w", header=True)
            self.log.info("Data saved to %s", self.output_file)
        except Exception as e:
            self.log.error("Failed to save data to %s: %s", self.output_file, e)


# Example usage with command-line arguments
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Process job application emails."
    )  # 4.  A duplicate exists when *both* conditions are true
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

    args = arg_parser.parse_args()

    email_agent = JobApplicationEmailAgent(
        input_file=args.input_file, output_file=args.output_file
    )

    email_agent.process_messages()
    email_agent.eliminate_duplicates()
    email_agent.write_to_csv()
