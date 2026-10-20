"""
Standalone script to test EIOS API extraction functions (GPHIN, ProMED, DONs).

Since the Board IDs are empty, the functions will use filter_items with source filters
to test the inner API extraction logic directly.
"""

import time
import json
import datetime

# ============================================================
# Configuration
# ============================================================

# Leave empty to test direct filter_items extraction (no board-based queries)
EIOS_WhoDonsBoard_ID = ""
EIOS_PromedBoard_ID = ""
EIOS_GPHINBoard_ID = ""


# ============================================================
# Helper functions
# ============================================================

def get_EIOS_client(api_version: str = "1.0"):
    """Create and return an EIOSClient instance (handles token acquisition internally)."""
    import config
    from eios_client import EIOSClient
    return EIOSClient(
        tenant_id=config.eios_tenant_id,
        client_id=config.eios_client_id,
        client_secret=config.eios_client_secret,
        scope=config.eios_scope,
        api_version=api_version,
        timeout=1200,
    )


def eios_call_with_retry(func, max_retries=3, sleep_seconds=2):
    """Call an EIOS function with retry on failure. Sleeps and retries on exception."""
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            print(f"  [!] EIOS request failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"      Sleeping {sleep_seconds} seconds and retrying...")
                time.sleep(sleep_seconds)
            else:
                print(f"      Max retries reached. Skipping this request.")
                return None


def adapt_date_format(date_str):
    """Adapt date string to EIOS expected format."""
    if date_str.endswith('Z'):
        return date_str
    if '.' not in date_str:
        return date_str + ":00.000Z"
    return date_str + "Z"


# ============================================================
# Extraction functions
# ============================================================

def getPromedArticles(startdate_promed_ts_string, enddate_promed_ts_string, EIOS_PromedBoard_ID, eios_client):
    """Get ProMED articles from EIOS using the new client API.
    
    If EIOS_PromedBoard_ID is null or empty, uses filter_items with source filter
    instead of get_matching_board_items.
    """
    if not EIOS_PromedBoard_ID:
        query = {
            "rules": [
                {
                    "operator": "dateRange",
                    "property": "processedOnDate",
                    "value": {
                        "gte": adapt_date_format(startdate_promed_ts_string),
                        "lte": adapt_date_format(enddate_promed_ts_string)
                    },
                    "name": "filterDate"
                },
                {
                    "operator": "in",
                    "property": "source.id",
                    "value": [
                        "promed",
                        "promed-ahead",
                        "promed-edr",
                        "promed-mail",
                        "promed-mbd",
                        "promedmail.org-38"
                    ],
                    "name": "includeSourcesRule"
                }
            ],
            "operator": "and",
            "start": 0,
            "limit": 100,
            "sorts": [
                {
                    "property": "processedOnDate",
                    "direction": "desc"
                }
            ]
        }
        result = eios_call_with_retry(lambda: eios_client.filter_items(query))
    else:
        result = eios_call_with_retry(lambda: eios_client.get_matching_board_items(
            boardId=EIOS_PromedBoard_ID,
            time_since=startdate_promed_ts_string,
            time_until=enddate_promed_ts_string,
            limit=100,
        ))
    return result


def getDONSArticles(startdate_dons_ts_string, enddate_dons_ts_string, EIOS_WhoDonsBoard_ID, eios_client):
    """Get WHO DONs articles from EIOS using the new client API.
    
    If EIOS_WhoDonsBoard_ID is null or empty, uses filter_items with a source filter for WHODON.
    """
    if not EIOS_WhoDonsBoard_ID:
        query = {
            "rules": [
                {
                    "operator": "dateRange",
                    "property": "processedOnDate",
                    "value": {
                        "gte": adapt_date_format(startdate_dons_ts_string),
                        "lte": adapt_date_format(enddate_dons_ts_string)
                    },
                    "name": "filterDate"
                },
                {
                    "operator": "in",
                    "property": "source.id",
                    "value": ["WHODON"],
                    "name": "includeSourcesRule"
                }
            ],
            "operator": "and",
            "start": 0,
            "limit": 100,
            "sorts": [
                {
                    "property": "processedOnDate",
                    "direction": "desc"
                }
            ]
        }
        result = eios_call_with_retry(lambda: eios_client.filter_items(query))
    else:
        result = eios_call_with_retry(lambda: eios_client.get_matching_board_items(
            boardId=EIOS_WhoDonsBoard_ID,
            time_since=startdate_dons_ts_string,
            time_until=enddate_dons_ts_string,
            limit=100,
        ))
    return result


def getGPHINArticles(startdate_gphin_ts_string, enddate_gphin_ts_string, EIOS_GPHINBoard_ID, eios_client):
    """Get GPHIN articles from EIOS using the new client API.
    
    If EIOS_GPHINBoard_ID is null or empty, uses filter_items with a source filter for GPHIN.
    """
    if not EIOS_GPHINBoard_ID:
        query = {
            "rules": [
                {
                    "operator": "dateRange",
                    "property": "processedOnDate",
                    "value": {
                        "gte": adapt_date_format(startdate_gphin_ts_string),
                        "lte": adapt_date_format(enddate_gphin_ts_string)
                    },
                    "name": "filterDate"
                },
                {
                    "operator": "in",
                    "property": "source.id",
                    "value": ["GPHIN", "gphin"],
                    "name": "includeSourcesRule"
                }
            ],
            "operator": "and",
            "start": 0,
            "limit": 100,
            "sorts": [
                {
                    "property": "processedOnDate",
                    "direction": "desc"
                }
            ]
        }
        result = eios_call_with_retry(lambda: eios_client.filter_items(query))
    else:
        result = eios_call_with_retry(lambda: eios_client.get_matching_board_items(
            boardId=EIOS_GPHINBoard_ID,
            time_since=startdate_gphin_ts_string,
            time_until=enddate_gphin_ts_string,
            limit=100,
        ))
    return result


def getMedicalOfficialArticles(startdate_ts_string, enddate_ts_string, eios_client):
    """Get English Medical Official articles from EIOS.

    Confirmed working filter properties:
      source.subject     -> "Medical Official"
      source.type        -> "Medical"
      languageIso        -> "en" (article language)
      source.languageCode -> "en" (source language)
    """
    query = {
        "rules": [
            {
                "operator": "dateRange",
                "property": "processedOnDate",
                "value": {
                    "gte": adapt_date_format(startdate_ts_string),
                    "lte": adapt_date_format(enddate_ts_string)
                },
                "name": "filterDate"
            },
            {
                "operator": "in",
                "property": "source.subject",
                "value": ["Medical Official"],
                #"value": ["Medical Official", "Medical"],
                "name": "includeSourceSubjectsRule"
            },
            {
                "operator": "in",
                "property": "source.type",
                "value": ["Medical"],
                "name": "includeSourceTypesRule"
            },
            {
                "operator": "in",
                "property": "languageIso",
                "value": ["en"],
                "name": "includeLanguageRule"
            },
            {
                "operator": "in",
                "property": "source.languageCode",
                "value": ["en"],
                "name": "includeSourceLanguageRule"
            }
        ],
        "operator": "and",
        "start": 0,
        "limit": 100,
        "sorts": [
            {
                "property": "processedOnDate",
                "direction": "desc"
            }
        ]
    }
    result = eios_call_with_retry(lambda: eios_client.filter_items(query))
    return result

if __name__ == "__main__":

    # Sample date range for testing (adjust as needed)
    # Format must be "%Y-%m-%dT%H:%M" (no seconds) — adapt_date_format appends ":00.000Z"
    sample_start = "2023-05-05T00:00"
    sample_end = "2026-07-03T23:59"

    print("=" * 60)
    print("EIOS API Extraction Test")
    print(f"Date range: {sample_start} -> {sample_end}")
    print("=" * 60)

    # Initialize EIOS client
    print("\n[*] Initializing EIOS client...")
    eios_client = get_EIOS_client()
    print("    Client ready.\n")

    # # --- Filter values lookup (available metadata for filters) ---
    # print("-" * 60)
    # print("[TEST] get_filter_values_lookup")
    # print("-" * 60)
    # filter_values = eios_call_with_retry(lambda: eios_client.get_filter_values_lookup())
    # if filter_values:
    #     print(f"  Top-level keys: {list(filter_values.keys()) if isinstance(filter_values, dict) else type(filter_values)}")
    #     print(f"  Preview: {json.dumps(filter_values, default=str)[:50]}...")
    #     # Save full response to disk
    #     with open("eios_api_metadata.json", "w", encoding="utf-8") as f:
    #         json.dump(filter_values, f, default=str, indent=2)
    #     print(f"  Full response saved to: eios_api_metadata.json")
    # else:
    #     print("  No result / request failed.")

    # # --- Test ProMED ---
    # print("-" * 60)
    # print("[TEST] getPromedArticles")
    # print("-" * 60)
    # promed_result = getPromedArticles(sample_start, sample_end, EIOS_PromedBoard_ID, eios_client)
    # if promed_result:
    #     print(f"  Articles returned: {len(promed_result.get('items', promed_result.get('data', [])))} (top-level keys: {list(promed_result.keys()) if isinstance(promed_result, dict) else type(promed_result)})")
    #     print(f"  Sample (first 500 chars): {json.dumps(promed_result, default=str)[:500]}")
    # else:
    #     print("  No result / request failed.")

    # # --- Test DONs ---
    # print("\n" + "-" * 60)
    # print("[TEST] getDONSArticles")
    # print("-" * 60)
    # dons_result = getDONSArticles(sample_start, sample_end, EIOS_WhoDonsBoard_ID, eios_client)
    # if dons_result:
    #     articles = dons_result.get('result', [])
    #     print(f"  Articles returned: {len(articles)} | count: {dons_result.get('count')} (top-level keys: {list(dons_result.keys())})")
    #     print(f"  Sample (first 500 chars): {json.dumps(dons_result, default=str)[:500]}")
    # else:
    #     print("  No result / request failed.")


    # --- Test Medical Official ---
    print("\n" + "-" * 60)
    print("[TEST] getMedicalOfficialArticles")
    print("-" * 60)
    medical_result = getMedicalOfficialArticles(sample_start, sample_end, eios_client)
    if medical_result:
        articles = medical_result.get('result', [])
        print(f"  Articles returned: {len(articles)} | count: {medical_result.get('count')} (top-level keys: {list(medical_result.keys())})")
        print(f"  Sample (first 500 chars): {json.dumps(medical_result, default=str)[:500]}")
    else:
        print("  No result / request failed.")

   

    # # --- Test GPHIN ---
    # print("\n" + "-" * 60)
    # print("[TEST] getGPHINArticles")
    # print("-" * 60)
    # gphin_result = getGPHINArticles(sample_start, sample_end, EIOS_GPHINBoard_ID, eios_client)
    # if gphin_result:
    #     articles = gphin_result.get('result', [])
    #     print(f"  Articles returned: {len(articles)} | count: {gphin_result.get('count')} (top-level keys: {list(gphin_result.keys())})")
    #     print(f"  Sample (first 500 chars): {json.dumps(gphin_result, default=str)[:500]}")
    # else:
    #     print("  No result / request failed.")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
