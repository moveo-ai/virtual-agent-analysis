import base64
import os
import time

import pandas as pd
import requests
import utils.dialog_flow_analysis.common as common_utils
from dotenv import load_dotenv
from tqdm import tqdm
from utils.logger import logger

load_dotenv()

SUBDOMAIN = os.environ.get("ZENDESK_SUBDOMAIN")
USERNAME = os.environ.get("ZENDESK_USERNAME")
API_TOKEN = os.environ.get("ZENDESK_API_TOKEN")
TIMEOUT = int(os.environ.get("ZENDESK_API_TIMEOUT", 8000))


def encode_base64(input_string):
    """
    Encode input string to base64.

    Args:
        input_string (str): String to be encoded.

    Returns:
        str: Encoded string.
    """
    try:
        input_bytes = input_string.encode("utf-8")
        base64_bytes = base64.b64encode(input_bytes)
        base64_string = base64_bytes.decode("utf-8")
        return base64_string
    except UnicodeEncodeError as uee:
        logger.error(
            "Failed to encode string due to UnicodeEncodeError",
            input_string=input_string,
            error_message=str(uee),
        )
        return ""
    except TypeError as te:
        logger.error(
            "Failed to encode string due to TypeError",
            input_string=input_string,
            error_message=str(te),
        )
        return ""


def get_tags(ticket_id: str):
    """
    Get tags associated with a Zendesk ticket.

    Args:
        ticket_id (str): ID of the Zendesk ticket.

    Returns:
        list: List of tags associated with the ticket.
    """
    if not SUBDOMAIN or not USERNAME or not API_TOKEN:
        logger.error("Zendesk credentials not found in environment variables.")
        return []

    auth_token = encode_base64(f"{USERNAME}/token:{API_TOKEN}")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "moveo-notebook-analysis",
        "Authorization": f"Basic {auth_token}",
    }

    url = f"https://{SUBDOMAIN}.zendesk.com/api/v2/tickets/{ticket_id}/tags.json"

    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        response_json = response.json()
        return response_json["tags"]
    except requests.RequestException as e:
        logger.error(
            "Failed to fetch tags for ticket", ticket_id=ticket_id, error_message=str(e)
        )
        return []


def match_zendesk_id_to_session_id(csv_fname: str, session_csv_fname: str):
    """
    Match Zendesk ticket IDs to Moveo session IDs and save to CSV.

    Args:
        csv_fname (str): Filename of the CSV file containing Zendesk ticket data.
        output_fname (str): Filename to save the output CSV file.

    Returns:
        None
    """
    session_csv_file_path = os.path.join(common_utils.DATA_DIR, session_csv_fname)

    try:
        df = common_utils.load_data_from_csv(
            os.path.join(common_utils.DATA_DIR, csv_fname)
        )

        if os.path.exists(
            session_csv_file_path
        ) and not common_utils.should_override_existing_output_file(session_csv_fname):
            return

        session_ids = []

        tickets_ids = df["ChatEvaluationTicketID"]
        for ticket_id in tqdm(
            tickets_ids, total=len(tickets_ids), desc="Processing tickets"
        ):
            tags = get_tags(ticket_id=ticket_id)
            moveo_session_ids = [
                tag for tag in tags if tag.startswith("moveo_session_id_")
            ]

            if len(moveo_session_ids) != 1:
                logger.warning(
                    "Found multiple or no session IDs for ticket",
                    ticket_id=ticket_id,
                    session_ids=moveo_session_ids,
                )
                session_ids.append(common_utils.SESSION_NOT_FOUND)
                continue

            session_id = moveo_session_ids[0][len("moveo_session_id_") :]
            session_ids.append(session_id)
            time.sleep(0.1)

        df["SessionID"] = session_ids
        logger.info("Processed tickets successfully")
        logger.debug("Preview of DataFrame", df_preview=df.head())
        df.to_csv(session_csv_file_path, index=False)
        logger.info("Saved DataFrame to CSV file", output_fname=session_csv_fname)
    except Exception as e:
        logger.error("An error occurred while processing tickets", exc_info=e)


def get_output_file_path(csv_fname: str, suffix: str) -> str:
    """
    Get the output file path based on the input CSV filename and suffix.

    Args:
        csv_fname (str): Filename of the CSV file.
        suffix (str): Suffix to be appended to the filename.

    Returns:
        str: Output file path.
    """
    filename = os.path.splitext(csv_fname)[0]
    return os.path.join(common_utils.DATA_DIR, f"{filename}_{suffix}.csv")


def split_rows(df: pd.DataFrame, brain_name: str):
    """
    Split rows into three categories: virtual assistant, agents, and no agent name.

    Args:
        df (pd.DataFrame): DataFrame containing ticket data.
        brain_name (str): Name of the virtual assistant (brain).

    Returns:
        pd.DataFrame: DataFrames for virtual assistant, agents, and no AgentName.
    """
    virtual_assistant_rows = []
    agents_rows = []
    no_agent_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Splitting tickets"):
        if isinstance(row["AgentName"], str) and brain_name in row["AgentName"]:
            virtual_assistant_rows.append(row)
        elif pd.isnull(row["AgentName"]):
            no_agent_rows.append(row)
        elif not pd.isnull(row["AgentName"]):
            agents_rows.append(row)

    return (
        pd.DataFrame(virtual_assistant_rows),
        pd.DataFrame(agents_rows),
        pd.DataFrame(no_agent_rows),
    )


def split_agents_and_brain_tickets(csv_fname: str, brain_name: str):
    """
    Split tickets into three categories: virtual assistant, agents, and no agent name,
    based on the agent's name.

    Args:
        csv_fname (str): Filename of the CSV file containing ticket data.
        brain_name (str): Name of the virtual assistant (brain).

    Returns:
        None
    """
    try:
        df = pd.read_csv(os.path.join(common_utils.DATA_DIR, csv_fname))

        if df.empty:
            raise ValueError(f"No data found in {csv_fname}")

        virtual_assistant_csv_file_path = get_output_file_path(
            csv_fname, "virtual_assistant"
        )
        agents_csv_file_path = get_output_file_path(csv_fname, "agents")
        no_agent_csv_file_path = get_output_file_path(csv_fname, "no_agent_name")

        if all(
            os.path.exists(file_path)
            and not common_utils.should_override_existing_output_file(file_path)
            for file_path in [
                virtual_assistant_csv_file_path,
                agents_csv_file_path,
                no_agent_csv_file_path,
            ]
        ):
            return

        df_virtual_assistant, df_agents, df_no_agent = split_rows(df, brain_name)

        df_virtual_assistant.to_csv(virtual_assistant_csv_file_path, index=False)
        df_agents.to_csv(agents_csv_file_path, index=False)
        df_no_agent.to_csv(no_agent_csv_file_path, index=False)

        logger.info("Split and saved DataFrame to CSV files.")

        total_rows_output_files = (
            len(df_virtual_assistant) + len(df_agents) + len(df_no_agent)
        )
        if total_rows_output_files == len(df):
            logger.info("Total rows in output files match total rows in input file.")
        else:
            logger.warning(
                "Total rows in output files do not match total rows in input file."
            )

    except FileNotFoundError:
        logger.error(f"File {csv_fname} not found")
    except ValueError as ve:
        logger.error(f"An error occurred while splitting tickets: {ve}")
    except Exception as e:
        logger.error("An error occurred while splitting tickets:", exc_info=e)
