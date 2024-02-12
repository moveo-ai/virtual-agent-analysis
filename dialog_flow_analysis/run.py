import requests
import base64
import pandas as pd
from tqdm import tqdm
import time
import structlog
from python_graphql_client import GraphqlClient
import os
from ast import literal_eval
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px


logger = structlog.get_logger(__name__)

###### Moveo Information #####
MOVEO_API_KEY = "xxxxx"
MOVEO_ACCOUNT_ID = "xxxxxx"
##################

##### Zendesk Information #########
SUBDOMAIN = "xxxxxx"
# generate one by following https://support.zendesk.com/hc/en-us/articles/4408889192858-Managing-access-to-the-Zendesk-API#topic_tcb_fk1_2yb
ZENDESK_API_TOKEN = "xxxxxxx"
USERNAME = "xxxxxxx"
##################

SESSION_NOT_FOUND = "NaN"


def encode_base64(input_string):
    input_bytes = input_string.encode("utf-8")
    base64_bytes = base64.b64encode(input_bytes)
    base64_string = base64_bytes.decode("utf-8")
    return base64_string


def get_tags(ticket_id: str):
    auth_token = encode_base64(f"{USERNAME}/token:{ZENDESK_API_TOKEN}")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "moveo-notebook-analysis",
        "Authorization": f"Basic {auth_token}",
    }

    url = f"https://{SUBDOMAIN}.zendesk.com/api/v2/tickets/{ticket_id}/tags.json"

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response_json = response.json()
    return response_json["tags"]


def load_data_from_file(csv_file_path: str):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    return df


def match_zendesk_id_to_session_id(csv_fname: str, output_fname: str):
    df = load_data_from_file(csv_fname)

    if os.path.exists(output_fname):
        print(
            f"WARNING: output file {output_fname} already exists and will be overwritten"
        )

    session_ids = []
    tickets_ids = df["ChatEvaluationTicketID"]
    for ticket_id in tqdm(tickets_ids, total=len(tickets_ids)):
        tags = get_tags(ticket_id=ticket_id)
        moveo_session_ids = [tag for tag in tags if tag.startswith("moveo_session_id_")]
        if len(moveo_session_ids) != 1:
            print(
                f"WARNING: found list with more than a single session id for ticket {ticket_id}: f{moveo_session_ids}"
            )
            session_ids.append(SESSION_NOT_FOUND)
            continue

        session_id = moveo_session_ids[0][len("moveo_session_id_") :]
        session_ids.append(session_id)
        time.sleep(0.1)

    df["SessionID"] = session_ids
    print(df.head())
    df.to_csv(output_fname, index=False)


def fetch_data_analytics_api(session_id, account_id, api_key):
    # logger.info(
    #     f"Fetching the data from the Analytics API for session_id='{session_id}'"
    # )

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
            print(
                "Received a 404 from the logs API. This is probably because the session does not exist"
            )
            return []
        print(
            "Could not fetch session content for session {session_id} Got error: {str(e)}"
        )
        return []

    # in case of error, raw_logs is the error message
    if errors := raw_logs.get("errors"):
        print(f"unable to get raw logs: {errors}")
        return []

    # logger.info("Fetched {} logs successfully".format(len(raw_logs["data"]["rows"])))

    return raw_logs["data"]["rows"]


def extract_flows_from_session_id(csv_file_path: str):
    df = load_data_from_file(csv_file_path)

    session_content = []
    for session_id in tqdm(df["SessionID"], total=len(df["SessionID"])):
        if session_id != SESSION_NOT_FOUND:
            content = fetch_data_analytics_api(
                session_id, MOVEO_ACCOUNT_ID, MOVEO_API_KEY
            )
        else:
            content = []

        session_content.append(content)
        time.sleep(0.1)

    df["SessionContent"] = session_content
    csv_file_path = "data/account_issues_br_content.csv"
    df.to_csv(csv_file_path, index=False)


def analyze_flows(csv_file_path):
    df = load_data_from_file(csv_file_path)
    df["SessionContent"] = df["SessionContent"].apply(literal_eval)

    contained = []
    flows = []
    for c in df["SessionContent"]:
        is_contained = False
        nodes_visited = []
        if len(c) >= 1:
            metadata = c[0]
            is_contained = metadata.get("is_contained")

            messages_metadata = metadata.get("messages", [])
            if not messages_metadata:
                print("Waring: NO MESSAGES PRESENT")
            else:
                brain_send_message = dict()
                for message in reversed(messages_metadata):
                    if message.get("event") == "message:brain_send":
                        brain_send_message = message
                        break

                # print("Last brain send , ", brain_send_message)
                nodes_stack = (
                    brain_send_message.get("message", {})
                    .get("debug", {})
                    .get("nodes_stack")
                )
                nodes_visited = [n.get("name") for n in nodes_stack]

        contained.append(is_contained)
        flows.append(nodes_visited)

    df["Contained"] = contained
    df["Flows"] = flows

    detractors = df[df["Question1"] <= 0]
    unique_detractors = detractors["UserName"].nunique()
    print("Duplicate detractors: ", len(detractors) - unique_detractors)

    neutral = df[(df["Question1"] > 3) & (df["Question1"] < 7)]
    promoters = df[df["Question1"] >= 7]
    unique_promoters = promoters["UserName"].nunique()
    print("Duplicate promoters: ", len(promoters) - unique_promoters)

    # account_locked_detractors = count_flow_name(detractors, "Account Locked")
    # forgot_password_detractors = count_flow_name(detractors, "Forgot Password")
    # account_reactivation_detractors = count_flow_name(
    #     detractors, "Account_Reactivation"
    # )
    # forgot_email_detractors = count_flow_name(detractors, "Forgot Email")

    # print("Account Locked Detractors: ", account_locked_detractors)
    # print("forgot_password Detractors: ", forgot_password_detractors)
    # print("account_reactivation Detractors: ", account_reactivation_detractors)
    # print("forgot_email Detractors: ", forgot_email_detractors)

    # ### Promoters ####
    # account_locked_promoters = count_flow_name(promoters, "Account Locked")
    # forgot_password_promoters = count_flow_name(promoters, "Forgot Password")
    # account_reactivation_promoters = count_flow_name(promoters, "Account_Reactivation")
    # forgot_email_promoters = count_flow_name(promoters, "Forgot Email")

    # print("Account Locked promoters: ", account_locked_promoters)
    # print("forgot_password promoters: ", forgot_password_promoters)
    # print("account_reactivation promoters: ", account_reactivation_promoters)
    # print("forgot_email promoters: ", forgot_email_promoters)

    # create_sankey(detractors, title="Detractors Journey Flow (Ratings: 0-3)")
    # create_sankey(promoters, title="Promoters Journey Flow (Ratings: 7-10)")
    # create_sankey(neutral, title="Neutrals Journey Flow (Ratings: 4-6)", topK=3)

    # plot_histogram(df)

    ############ KEY STATS ############
    num_detractors = len(detractors)
    print("Total Ratings: ", len(df))
    print("Number of Detractors: ", len(detractors))
    print("Number of Neutrals: ", len(neutral))
    print("Number of Promoters: ", len(promoters))
    print("Average CSAT: ", df["Question1"].mean())
    print("Median CSAT: ", df["Question1"].median())

    ####

    # print("########################")
    # filtered_all = df[df["Flows"].apply(lambda x: len(x) > 0)]
    # print("Filtered Empty CSAT Mean", filtered_all["Question1"].mean())
    # print("Filtered Empty CSAT Median", filtered_all["Question1"].median())

    # filter_out = ["Unknown", "Handover"]
    # filtered_all = filter_out_subflows(filtered_all, filter_out)
    # print("Filtered Empty CSAT Mean", filtered_all["Question1"].mean())
    # print("Filtered Empty CSAT Median", filtered_all["Question1"].median())

    # filter_out = ["Account Locked", "Not logged in"]
    # filtered_all = filter_out_subflows(filtered_all, filter_out)
    # print("Filtered Empty CSAT Mean", filtered_all["Question1"].mean())
    # print("Filtered Empty CSAT Median", filtered_all["Question1"].median())
    # print("#########################")

    # for session_id, flow in zip(
    #     detractors["ChatEvaluationTicketID"], detractors["Flows"]
    # ):
    #     for f in flow:
    #         if f == "Account Locked":
    #             print("sess id: ", session_id)

    # exit(1)

    ###


def create_sankey(df: pd.DataFrame, title: str, topK=10):
    flows = df["Flows"]
    transitions = []
    for journey in flows:
        for i in range(len(journey) - 1):
            transitions.append((journey[i], journey[i + 1]))

    transition_counts = Counter(transitions)
    # some pairs are fobidden due to common sense (maybe the session expired there)
    forbidden_flows = [
        ("AB test Passed", "Default Message Start"),
        ("AB test Passed", "Greetings"),
        ("Handover", "Handover select department"),
        ("Check DOB", "Check DOB"),
    ]
    for f in forbidden_flows:
        if f in transition_counts:
            transition_counts[f] = -1

    transition_counts = Counter(
        {item: count for item, count in transition_counts.items() if count > topK}
    )

    # Unique states
    unique_states = set(
        [state for transition in transition_counts.keys() for state in transition]
    )

    # Map each state to an index
    state_to_index = {state: idx for idx, state in enumerate(unique_states)}

    # Prepare source, target, and value lists
    sources = []
    targets = []
    values = []

    print("transition counts: ", transition_counts)

    for (source, target), count in transition_counts.items():
        if count > 10:
            sources.append(state_to_index[source])
            targets.append(state_to_index[target])
            values.append(count)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=list(unique_states),
                ),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )

    fig.update_layout(
        title_text=f"<b>{title}</b>",  # Bold text using HTML tags
        title_font=dict(size=40),  # Adjust font size as needed
        title_x=0.5,  # Center the title by setting its x position to 0.5
    )

    fig.show()


def count_flow_name(df: pd.DataFrame, flow_name: str):
    """Count number of times a flow occurs accross all rows of Flows"""
    return df["Flows"].apply(lambda x: x.count(flow_name)).sum()


def plot_histogram(df):
    # Plotting the histogram using Plotly
    fig = px.histogram(df, x="Question1", nbins=11, title="Histogram of Ratings Values")
    fig.update_layout(bargap=0.1)
    fig.update_traces(
        marker_color="blue", marker_line_color="black", marker_line_width=1.5
    )
    fig.update_xaxes(title_text="Value", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Frequency")
    fig.show()
    skewness = df["Question1"].skew()
    print(f"Skewness: {skewness}")


def get_node_counts(df):
    all_nodes_detractors = [string for sublist in df["Flows"] for string in sublist]
    return Counter(all_nodes_detractors)
    print("nodes counts: ", node_counts)


def main():
    # input_fname = "data/account_issues_br.csv"
    # output_fname = "data/account_issues_br_session_id.csv"

    # input_fname = "data/kyc_br.csv"
    # output_fname = "data/kyc_br_session_id.csv"
    # match_zendesk_id_to_session_id(input_fname, output_fname)

    # extract_flows_from_session_id("data/account_issues_br_session_id.csv")

    analyze_flows("data/account_issues_br_content.csv")


if __name__ == "__main__":
    main()
