"""Small REST client for a Virtuoso SPARQL endpoint.

Used by the entity-linking step of ``ONTO-RESOURCES_eKG`` to resolve disease and place terms
against the ontologies loaded in the triple store. Responses are cached on disk with joblib so that
repeated runs do not re-issue identical queries. HTTP Digest authentication is tried first, with a
fallback to Basic authentication.
"""

import os
import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
import ssl
import json

from joblib import Memory

cachedir = 'cached'
mem = Memory(cachedir, verbose=False)

# Seconds allowed to a single SPARQL request. It is generous, because a fuzzy bif:contains query on
# the whole KG collection can legitimately take minutes, but it is there so that an endpoint which
# stops answering cannot block the pipeline for ever.
SPARQL_REQUEST_TIMEOUT = 300
# The connection itself must open quickly: an unreachable endpoint otherwise blocks every query for
# ~130s (OS TCP connect timeout) before failing. Passed to requests as (connect, read) timeouts.
SPARQL_CONNECT_TIMEOUT = 10

# When a corporate proxy is configured in the environment but the Virtuoso host happens to be
# listed in no_proxy, requests would bypass the proxy and fail to reach the endpoint. Since proxies
# passed explicitly are always used, for the Virtuoso calls we pass the environment proxy explicitly
# (equivalent to curl --noproxy ""). Without a proxy in the environment this returns None and
# requests behaves normally.
def virtuoso_proxies():
    proxies = {}
    for scheme in ("http", "https"):
        proxy = os.environ.get(f"{scheme}_proxy") or os.environ.get(f"{scheme.upper()}_PROXY")
        if proxy:
            proxies[scheme] = proxy
    return proxies or None



@mem.cache
def execute_query(endpoint, query, auth):
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/sparql-results+json'
    }

    params = {
        'query': query
    }

    response = requests.post(endpoint, headers=headers, params=params, auth=auth,
                             timeout=(SPARQL_CONNECT_TIMEOUT, SPARQL_REQUEST_TIMEOUT),
                             proxies=virtuoso_proxies())

    if response.status_code != 200:
        raise Exception(f"Failed to execute query: {response.text}")

    return response.text


def execute_query_no_cache(endpoint, query, auth):
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/sparql-results+json'
    }

    params = {
        'query': query
    }

    response = requests.post(endpoint, headers=headers, params=params, auth=auth,
                             timeout=(SPARQL_CONNECT_TIMEOUT, SPARQL_REQUEST_TIMEOUT),
                             proxies=virtuoso_proxies())

    if response.status_code != 200:
        raise Exception(f"Failed to execute query: {response.text}")

    return response.text


def sparqlQuery(endpoint, query, usernameVirt, passwordVirt, USE_CACHE=True):
    """
    Make a SPARQL query to a Virtuoso SPARQL endpoint.

    Args:
    - endpoint (str): The URL of the Virtuoso SPARQL endpoint.
    - query (str): The SPARQL query to execute.
    - username (str): The username for authentication.
    - password (str): The password for authentication.

    Returns:
    - responseText (requests.Response): The responseText from the Virtuoso SPARQL endpoint.
    """

    # Use SSL context to establish a secure connection
    ssl_context = ssl.create_default_context()

    # Try HTTP Digest authentication
    try:
        auth = HTTPDigestAuth(usernameVirt, passwordVirt)
        if USE_CACHE:
            responseText = execute_query(endpoint, query, auth)
        else:
            responseText = execute_query_no_cache(endpoint, query, auth)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        # endpoint not reachable: retrying with Basic Auth would only wait for the same timeout again
        print(f"Virtuoso endpoint not reachable: {str(e)}")
        return None
    except Exception as e:
        print(f"HTTP Digest authentication failed: {str(e)}")

        # Fallback to Basic Auth
        try:
            auth = HTTPBasicAuth(usernameVirt, passwordVirt)
            if USE_CACHE:
                responseText = execute_query(endpoint, query, auth)
            else:
                responseText = execute_query_no_cache(endpoint, query, auth)
        except Exception as e:
            print(f"Basic Auth failed: {str(e)}")
            return None

    return responseText


#
# def sparqlQuery(endpoint, query, usernameVirt, passwordVirt):
#     """
#     Make a SPARQL query to a Virtuoso SPARQL endpoint.
#
#     Args:
#     - endpoint (str): The URL of the Virtuoso SPARQL endpoint.
#     - query (str): The SPARQL query to execute.
#     - username (str): The username for authentication.
#     - password (str): The password for authentication.
#
#     Returns:
#     - response (requests.Response): The response from the Virtuoso SPARQL endpoint.
#     """
#     headers = {
#         'Content-Type': 'application/x-www-form-urlencoded',
#         'Accept': 'application/sparql-results+json'
#     }
#
#     params = {
#         'query': query
#     }
#
#     # Use SSL context to establish a secure connection
#     ssl_context = ssl.create_default_context()
#
#     # Try HTTP Digest authentication
#     try:
#         auth = HTTPDigestAuth(usernameVirt, passwordVirt)
#         response = requests.post(endpoint, headers=headers, params=params, auth=auth,
#                             timeout=SPARQL_REQUEST_TIMEOUT)
#     except Exception as e:
#         print(f"HTTP Digest authentication failed: {str(e)}")
#
#         # Fallback to Basic Auth
#         try:
#             auth = HTTPBasicAuth(usernameVirt, passwordVirt)
#             response = requests.post(endpoint, headers=headers, params=params, auth=auth,
#                             timeout=SPARQL_REQUEST_TIMEOUT)
#         except Exception as e:
#             print(f"Basic Auth failed: {str(e)}")
#             return None
#
#     if response.status_code != 200:
#         raise Exception(f"Failed to execute query: {response.text}")
#
#     return response


if __name__ == '__main__':
    # Example usage
    endpoint = 'https://api-vast.jrc.service.ec.europa.eu/sparql'

    VirtuosoUsername = 'dba'
    VirtuosoPassword = ''
    Virtuosokey_filename = 'VIRTUOSO-dba.key'

    USE_CACHE = False # True or False

    #############


    #choices = ['SNOMED', 'LOINC', 'ICD10', 'MESH', 'NCIT']  # restricts the input to these values only
    choices = ["AI", "AIO", "AEO", "BFO", "BIM", "BCGO", "CL", "CHIRO", "CHEBI", "DCM", "FMA", "GO", "GENO",
             "GeoSPARQL", "HL7", "DOID", "HP", "HP_O", "IDO", "IAO", "ICD10", "LOINC", "MESH",
             "MONDO", "NCIT", "NCBITAXON", "NCBITaxon_", "NIFCELL", "NIFSTD", "GML", "OBCS", "OCHV", "OHPI",
             "OPB", "TRANS", "PLOSTHES", "RADLEX", "RO", "STY", "SO", "SNOMED", "STATO",
             "SYMP", "FoodOn", "UBERON", "ORDO", "HOOM", "VO", "OGMS", "EuroSciVoc", "ARO", "REACTO", "go-lego", "go-lego-reacto", "PR", "PSIMOD", "pathway_like_go_cams", "ECMO"]

    # Construct the FROM clauses
    from_clauses = ' '.join([f"FROM <{choice}>" for choice in choices])

    #word = "acute sinusitis"
    word = "pure mathematics"
    # Construct the full SPARQL query
    query = f"""
            prefix skosxl: <http://www.w3.org/2008/05/skos-xl#>
            SELECT ?concept ?label (COUNT(?edge) AS ?score)
            {from_clauses}
            WHERE {{
              ?concept skos:prefLabel|rdfs:label|skos:altLabel|skosxl:literalForm|obo:hasRelatedSynonym ?label .
              FILTER (LCASE(STR(?label)) = "{word.lower()}")
              ?concept ?edge ?o .
            }}
            GROUP BY ?concept ?label
            ORDER BY DESC(?score)
        """

    print(query)

    ###############

    if Virtuosokey_filename:
        fkeyname = Virtuosokey_filename
        with open(fkeyname) as f:
            VirtuosoPassword = f.read().strip()

    responseText = sparqlQuery(endpoint, query, VirtuosoUsername, VirtuosoPassword, USE_CACHE)

    # Parse the response as JSON

    results = json.loads(responseText)

    # Print the results
    if len(results) > 0 and results['results']['bindings']:
        for result in results['results']['bindings']:
            print(result)
    else:
        print("!!! VIRTUOSO NO RESULTS !!!")