from typing import Dict, List

import requests
from python_graphql_client import GraphqlClient
from utils.logger import logger


def initialize_graphql_clinet(account_id: str, api_key: str) -> GraphqlClient:
    """
    Initializes a Graph QL client

    Args:
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

    return client


def execute_query(client: GraphqlClient, query: str, variables: Dict) -> list:
    """
    Fetch data from the Analytics API for a given session ID.

    Args:
        client (GraphqlClient): GraphQL client
        query (str): The GraphQL query. More information:
            https://docs.moveo.ai/docs/analytics/api_overview
        variables (dict): A dictionary with key = variable name of the query
            and value = the variable value

    Returns:
        list: List of logs retrieved from the API.
    """

    try:
        raw_logs: dict = client.execute(
            query=query,
            variables=variables,
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning("Received a 404 from the GrapgQL API", variables=variables)
            return []
        logger.error(
            f"Could not fetch content from the GraphQL API. Got error: {str(e)}",
            variables=variables,
        )
        return []
    except ValueError as e:
        # Handle the JSON parsing error
        logger.error(
            f"Error parsing JSON response from the GraphQL API. Got error: {e}",
            variables=variables,
        )
        return []

    # in case of error, raw_logs is the error message
    if errors := raw_logs.get("errors"):
        logger.error(
            f"Unable to get result from GraphQL API. Got errors: {errors}",
            variables=variables,
        )
        return []

    return raw_logs["data"]["rows"]


def serialize_list_variable(string_list: List[str]) -> str:
    """
    Returns a serialized version of the input list of strings to be passed
    as GraphQL variables
    Args:
        string_list (List[str]): the input strings
    Returns:
        a serialized string that is compatible with GraphQL
    """
    # Format each string in the list to be double-quoted
    formatted_strings = [f'"{s}"' for s in string_list]
    # Join the formatted strings with a comma and space
    # and enclose them in curly braces
    return "{" + ", ".join(formatted_strings) + "}"
