import os
import time
from ast import literal_eval
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from python_graphql_client import GraphqlClient
from tqdm import tqdm
from utils.logger import logger

load_dotenv()

# Used to get the sessions from analytics
MOVEO_API_KEY = os.environ.get("MOVEO_API_KEY")
MOVEO_ACCOUNT_ID = os.environ.get("MOVEO_ACCOUNT_ID")

# Used to create the diagrams
DETRACTORS_UPPER_LIMIT = int(os.environ.get("DETRACTORS_UPPER_LIMIT", 3))
PROMOTERS_LOWER_LIMIT = int(os.environ.get("PROMOTERS_LOWER_LIMIT", 7))
RATING_MIN = int(os.environ.get("RATING_MIN", 0))
RATING_MAX = int(os.environ.get("RATING_MAX", 10))

# As a null value for session id
SESSION_NOT_FOUND = "No session_id"
SESSION_ID_LEN = 36

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"


def load_data_from_csv(csv_fname: str, **kwargs) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        csv_fname (str): The name of the CSV file.
        **kwargs: Additional keyword arguments to pass to pandas.read_csv.

    Returns:
        pandas.DataFrame: The loaded data as a DataFrame.
    """
    file_path = os.path.join(DATA_DIR, csv_fname)
    try:
        # Load the CSV file into a pandas DataFrame with additional keyword arguments
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError as fnf_error:
        logger.error("File not found", file_path=file_path, error=str(fnf_error))
        return None
    except Exception as e:
        logger.error(
            "An error occurred while loading data",
            file_path=file_path,
            error=str(e),
        )
        return None


def fetch_data_analytics_api(session_id: str, account_id: str, api_key: str) -> list:
    """
    Fetch data from the Analytics API for a given session ID.

    Args:
        session_id (str): The ID of the session.
        account_id (str): The Moveo account ID.
        api_key (str): The API key for authentication.

    Returns:
        list: List of logs retrieved from the API.
    """
    client = GraphqlClient(
        endpoint="https://logs.moveo.ai/v1/graphql",
        headers={
            "Authorization": f"apikey {api_key}",
            "X-Moveo-Account-Id": account_id,
        },
    )

    query = """
            query SessionContentV2($sessionId: String) {
                rows: log_session_content_v2(args: { session_id: $sessionId }) {
                    messages
                    brain_id
                    brain_parent_id
                    avg_confidence
                    brain_version
                    channel
                    channel_user_id
                    desk_id
                    end_time
                    external_user_id
                    integration_id
                    is_contained
                    is_covered
                    is_test
                    min_confidence
                    participated_agents
                    rating
                    session_id
                    start_time
                    tags
                    total_user_messages
                    user_id
                    user_name
                }
            }
        """

    try:
        raw_logs: dict = client.execute(
            query=query,
            variables={
                "sessionId": session_id,
            },
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(
                f"Received a 404 from the logs API for session {session_id}. "
                "This is probably because the session does not exist"
            )
            return []
        logger.error(
            f"Could not fetch content for session {session_id}. Got error: {str(e)}"
        )
        return []
    except ValueError as e:
        # Handle the JSON parsing error
        logger.error(f"Error parsing JSON response for session {session_id}: {e}")
        return []

    # in case of error, raw_logs is the error message
    if errors := raw_logs.get("errors"):
        logger.error(f"Unable to get raw logs for session {session_id}: {errors}")
        return []

    return raw_logs["data"]["rows"]


def extract_flows_from_session_id(
    sessions_csv_fname: str,
    content_csv_fname: str,
):
    """
    Extract flows from session IDs and save them to a CSV file.

    Args:
        sessions_csv_fname (str): The name of the CSV file containing session IDs.
        content_csv_fname (str): The name of the CSV file with session content that
        will be created.

    Returns:
        None
    """

    content_csv_file_path = os.path.join(DATA_DIR, content_csv_fname)

    if os.path.exists(
        os.path.join(content_csv_file_path)
    ) and not should_override_existing_output_file(content_csv_fname):
        return

    df = load_data_from_csv(sessions_csv_fname)

    session_content = []
    for session_id in tqdm(
        df["SessionID"], total=len(df["SessionID"]), desc="Extracting flows"
    ):
        if len(session_id) == SESSION_ID_LEN:
            content = fetch_data_analytics_api(
                session_id, MOVEO_ACCOUNT_ID, MOVEO_API_KEY
            )
        else:
            content = []

        session_content.append(content)
        time.sleep(0.1)

    df["SessionContent"] = session_content
    df.to_csv(content_csv_file_path, index=False)
    logger.info("Flows extracted and saved to CSV file", csv_file=content_csv_file_path)


def analyze_agents(content_csv_fname: str):
    """
    Analyzes agent data stored in a CSV file.

    Parameters:
    - content_csv_fname (str): The filename of the CSV file containing the agent data.

    Returns:
    - None

    This function loads the agent data from the specified CSV file,
    creates a DataFrame (df) and generates a histogram plot
    of the ratings of the agents.

    """
    # Load data from CSV file
    df = load_data_from_csv(content_csv_fname)

    plot_ratings_histogram(df, fname=content_csv_fname)

    return df


def analyze_flows(content_csv_fname: str, min_transitions_displayed=3):
    """
    Analyze flows from a CSV file containing session data.

    This function loads session data from a CSV file, extracts relevant information
    such as contained status and visited nodes, categorizes sessions based on rating,
    and generates Sankey diagrams, histograms and pie charts for visualization.

    Args:
        content_csv_fname (str): The name of the CSV file containing session data.

    Returns:
        None
    """
    # Load data from CSV file
    df = load_data_from_csv(content_csv_fname)

    # Convert SessionContent column to Python objects
    df["SessionContent"] = df["SessionContent"].apply(literal_eval)

    # Extract contained status and visited nodes
    contained = []
    covered = []
    flows = []
    for c in df["SessionContent"]:
        is_contained = False
        nodes_visited = []
        if len(c) >= 1:
            metadata = c[0]
            is_contained = metadata.get("is_contained")
            is_covered = metadata.get("is_covered")

            messages_metadata = metadata.get("messages", [])
            if not messages_metadata:
                logger.warn("NO MESSAGES PRESENT")
            else:
                brain_send_message = {}
                for message in reversed(messages_metadata):
                    if message.get("event") == "message:brain_send":
                        brain_send_message = message
                        break

                nodes_stack = (
                    brain_send_message.get("message", {})
                    .get("debug", {})
                    .get("nodes_stack")
                )
                nodes_visited = [n.get("name") for n in nodes_stack]

        contained.append(is_contained)
        covered.append(is_covered)
        flows.append(nodes_visited)

    # Add new columns to DataFrame
    df["Contained"] = contained
    df["Covered"] = covered
    df["Flows"] = flows

    # Categorize sessions based on rating
    detractors = detractors_group(df)
    neutral = neutral_group(df)
    promoters = promoters_group(df)

    # Create the folder to store the analysis diagrams
    get_or_create_plot_directory(content_csv_fname)

    # Generate Sankey diagrams and histograms for visualization
    create_sankey(
        detractors,
        title="Detractors Journey Flow "
        f"(Ratings: {RATING_MIN}-{DETRACTORS_UPPER_LIMIT})",
        fname=content_csv_fname,
        top_k=min_transitions_displayed,
    )
    create_sankey(
        promoters,
        title=f"Promoters Journey Flow (Ratings: {PROMOTERS_LOWER_LIMIT}-{RATING_MAX})",
        fname=content_csv_fname,
        top_k=min_transitions_displayed,
    )
    create_sankey(
        neutral,
        title="Neutrals Journey Flow (Ratings: "
        f"{DETRACTORS_UPPER_LIMIT + 1}-{PROMOTERS_LOWER_LIMIT - 1})",
        fname=content_csv_fname,
        top_k=min_transitions_displayed,
    )
    # Generate the ratings histogram
    plot_ratings_histogram(
        df,
        fname=content_csv_fname,
    )
    # Generate containment pie chart
    create_pie_charts(
        df,
        "Contained",
        [
            detractors_group,
            neutral_group,
            promoters_group,
        ],
        "Containment",
        fname=content_csv_fname,
    )
    # Generate coverage pie chart
    create_pie_charts(
        df,
        "Covered",
        [
            detractors_group,
            neutral_group,
            promoters_group,
        ],
        "Coverage",
        fname=content_csv_fname,
    )
    return df


# NOTE: top_k should be adjusted to get better results depending on the ammount of data.
def create_sankey(
    df: pd.DataFrame,
    title: str,
    fname: str = None,
    top_k=3,
):
    """
    Create a Sankey diagram from a DataFrame.

    This function generates a Sankey diagram visualization from the provided DataFrame,
    which contains flow data.

    Args:
        df (pd.DataFrame): DataFrame containing flow data.
        title (str): Title of the Sankey diagram.
        top_k (int, optional): Min number of transitions to include.
                Defaults to 3. For a large ammount of data,
                increase it in order to get readable diagrams.

    Returns:
        None
    """
    flows = df["Flows"]
    transitions: List[Tuple[str, str]] = []
    for journey in flows:
        for i in range(len(journey) - 1):
            transitions.append((journey[i], journey[i + 1]))

    # Count transition occurrences
    transition_counts = Counter(transitions)

    # Forbid certain transitions due to common sense
    forbidden_flows: List[Tuple[str, str]] = [
        ("AB test Passed", "Default Message Start"),
        ("AB test Passed", "Greetings"),
        ("Handover", "Handover select department"),
        ("Check DOB", "Check DOB"),
    ]
    for f in forbidden_flows:
        if f in transition_counts:
            transition_counts[f] = -1

    # Filter transitions by count
    transition_counts = Counter(
        {item: count for item, count in transition_counts.items() if count >= top_k}
    )

    # Get unique states
    unique_states = {
        state for transition in transition_counts.keys() for state in transition
    }

    # Map states to indices
    state_to_index = {state: idx for idx, state in enumerate(unique_states)}

    # Prepare source, target, and value lists
    sources = []
    targets = []
    values = []

    for (source, target), count in transition_counts.items():
        if count >= top_k:
            sources.append(state_to_index[source])
            targets.append(state_to_index[target])
            values.append(count)

    # Create Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 15,
                    "thickness": 20,
                    "line": {"color": "black", "width": 0.5},
                    "label": list(unique_states),
                },
                link={"source": sources, "target": targets, "value": values},
            )
        ]
    )

    # Set layout options
    fig.update_layout(
        title_text=f"<b>{title}</b>",  # Bold text using HTML tags
        title_font={"size": 40},  # Adjust font size as needed
        title_x=0.5,  # Center the title by setting its x position to 0.5
        autosize=True,
        height=1200,
    )

    # Show the Sankey diagram
    fig.show()
    # Export diagram as an html file
    if fname is not None:
        # Export diagram as an html file
        fig.write_html(f"{get_or_create_plot_directory(fname)}/{title}.html")


def count_flow_name(df: pd.DataFrame, flow_name: str):
    """Count number of times a flow occurs accross all rows of Flows"""
    return df["Flows"].apply(lambda x: x.count(flow_name)).sum()


def plot_ratings_histogram(df: pd.DataFrame, fname: str = None):
    title = "Histogram of Ratings Values"
    # Plotting the histogram using Plotly
    fig = px.histogram(df, x="Question1", nbins=11)
    fig.update_layout(
        margin_t=100,
        bargap=0.1,
        title_text=f"<b>{title}</b>",  # Bold text using HTML tags
        title_font={"size": 40},  # Adjust font size as needed
        title_x=0.5,  # Center the title by setting its x position to 0.5
    )
    fig.update_traces(
        marker_color="blue", marker_line_color="black", marker_line_width=1.5
    )
    fig.update_xaxes(title_text="Value", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Frequency")
    fig.show()
    if fname is not None:
        # Export diagram as an html file
        fig.write_html(f"{get_or_create_plot_directory(fname)}/{title}.html")


def create_pie_charts(
    df, metric: str, groups: list[callable], title: str, fname: str = None
):
    fig = make_subplots(
        rows=1,
        cols=len(groups),
        specs=[[{"type": "pie"}] * len(groups)],
        subplot_titles=[
            group.__name__.replace("_group", "").capitalize() for group in groups
        ],  # Use function names as titles
    )

    for i, group_func in enumerate(groups, start=1):
        group_df = group_func(df)  # Call the function to filter the DataFrame
        values = group_df[metric]
        true_count = values.value_counts().get(True, 0)
        false_count = len(group_df) - true_count
        total_count = len(group_df)

        # Calculate percentages
        true_percentage = (true_count / total_count) * 100
        false_percentage = (false_count / total_count) * 100

        # Create a DataFrame for the counts and percentages
        data = pd.DataFrame(
            {
                "Value": [metric, f"Not {metric}"],
                "Count": [true_count, false_count],
                "Percentage": [true_percentage, false_percentage],
            }
        )

        pie_fig = px.pie(
            data,
            values="Count",
            names="Value",
        )

        for trace in pie_fig.data:
            fig.add_trace(trace, row=1, col=i)

    fig.update_layout(
        title_text=f"<b>{title}</b>",  # Bold text using HTML tags
        title_font={"size": 40},  # Adjust font size as needed
        title_x=0.5,  # Center the title by setting its x position to 0.5
    )

    if fname is not None:
        # Export diagram as an html file
        fig.write_html(f"{get_or_create_plot_directory(fname)}/{title}.html")
    fig.show()


def get_node_counts(df):
    all_nodes_detractors = [string for sublist in df["Flows"] for string in sublist]
    return Counter(all_nodes_detractors)


def should_override_existing_output_file(output_fname: str) -> bool:
    """
    Check if the output file should be overridden.

    Args:
        output_fname (str): Filename of the output file.

    Returns:
        bool: True if the file should be overridden, False otherwise.
    """
    try:
        override = input(
            f"WARNING: Output file '{output_fname}' already exists."
            "Do you want to override it? (Y/n): "
        ).lower()
        if override != "y":
            logger.info("Aborting operation.")
            return False
        logger.info(f"Output file '{output_fname}' will be overwritten.")
        return True
    except Exception as e:
        logger.error(
            "An error occurred while checking if the output file should be overridden",
            exc_info=e,
        )
        return False


def detractors_group(df):
    return df[df["Question1"] <= DETRACTORS_UPPER_LIMIT]


def neutral_group(df):
    return df[
        (df["Question1"] > DETRACTORS_UPPER_LIMIT)
        & (df["Question1"] < PROMOTERS_LOWER_LIMIT)
    ]


def promoters_group(df):
    return df[df["Question1"] >= PROMOTERS_LOWER_LIMIT]


def get_or_create_plot_directory(fname: str):
    """
    Create a directory if it does not exist.

    Parameters:
    - fname (str): The filename.
    - DATA_DIR (str): The directory path where the new directory will be created.

    returns:
        the plot directory path
    """
    directory_name = f"{os.path.splitext(fname)[0]}_plots"
    complete_path = os.path.join(DATA_DIR, directory_name)
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)

    return complete_path
