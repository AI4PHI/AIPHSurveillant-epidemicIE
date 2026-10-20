#!/usr/bin/env python3
"""Test EIOS API credentials and basic connectivity."""

from eios_client import EIOSClient, EIOSClientError
import config
import json

def adapt_date_format(date_str):
    if date_str.endswith('Z'):
        return date_str
    if '.' not in date_str:
        return date_str + ":00.000Z"
    return date_str + "Z"

def main():
    print("=== EIOS API Credential Test ===\n")

    api_version = "1.0"
    print(f"--- Testing with api_version='{api_version}' ---\n")

    client = EIOSClient(
        tenant_id=config.eios_tenant_id,
        client_id=config.eios_client_id,
        client_secret=config.eios_client_secret,
        scope=config.eios_scope,
        api_version=api_version,
        timeout=1200,  #5 min
    )

    me = client.get_current_user_profile()
    print(me)

    # Test token
    try:
        client.acquire_token()
        print(f"1. Token: OK\n")
    except EIOSClientError as e:
        print(f"1. Token: FAILED - {e}\n")
        
    # Test authenticated endpoint
    print("2. Testing get_current_user_profile...")
    try:
        profile = client.get_current_user_profile()
        print(f"   OK: {profile}\n")
    except EIOSClientError as e:
        print(f"   FAILED: {e}\n")

    # Test hello_items endpoint (no auth required)
    print("3. Testing hello_items (no auth)...")
    try:
        result = client.hello_items()
        print(f"   OK: {result}\n")
    except EIOSClientError as e:
        print(f"   FAILED: {e}\n")
    
    # Test hello_boards endpoint (no auth required)
    print("4. Testing hello_boards (no auth)...")
    try:
        result = client.hello_boards()
        print(f"   OK: {result}\n")
    except EIOSClientError as e:
        print(f"   FAILED: {e}\n")

    # Test hello_posts endpoint (no auth required)
    print("5. Testing hello_posts (no auth)...")
    try:
        result = client.hello_posts()
        print(f"   OK: {result}\n")
    except EIOSClientError as e:
        print(f"   FAILED: {e}\n")

    # Test get_matching_board_items endpoint (authenticated)
    print("6. Testing get_matching_board_items (authenticated)...")
    try:
        board_id = "1776700887443" #digital-health_former-promed
        #board_id = "1776702435191" #digital-health_whodon
        #board_id = "1776878873024" #digital-health-Medical-Official-PH
        #board_id = "eios1-14714"
        #
        startdate_ts_string="2026-01-01T00:00"
        enddate_ts_string="2026-04-23T17:00"
        #
        result = client.get_matching_board_items(
            board_id,
            #time_since="now-28d/d",
            time_since=startdate_ts_string, #"2026-01-01T00:00",
            time_until=enddate_ts_string, #"2026-04-23T17:00",                
        )
        print(f"   OK: result={result}\n")
        if result and 'result' in result:
            print(f"   Found {len(result['result'])} items")
        if result and 'count' in result:
            print(f"   Total count: {result['count']}")
        print()
        
        # Store items for subsequent test
        items_result = result
    except EIOSClientError as e:
        print(f"   FAILED: {e}\n")
        items_result = None

    # Test get_item_details using an itemId from get_matching_board_items
    print("7. Testing get_item_details (authenticated)...")
    try:
        if items_result and 'result' in items_result and len(items_result['result']) > 0:
            # Extract itemId from the first item
            first_item = items_result['result'][0]
            item_id = first_item.get('id')
            if item_id:
                print(f"   Using itemId: {item_id}")
                
                # Call using production client
                details_result = client.get_item_details(item_id)
                print(f"   Production OK: result={details_result}\n")
                
                # Create a test client and call the same endpoint
                test_client = EIOSClient(
                    tenant_id=config.eios_tenant_id,
                    client_id=config.eios_client_id,
                    client_secret=config.eios_client_secret,
                    scope=config.eios_scope,
                    base_url=EIOSClient.DEFAULT_TEST_URL,
                    api_version=api_version,
                    timeout=1200,  #5 min
                )
                print(f"   Testing with test environment: {EIOSClient.DEFAULT_TEST_URL}")
                test_details_result = test_client.get_item_details(item_id)
                print(f"   Test OK: result={test_details_result}\n")
            else:
                print(f"   SKIPPED: No itemId found in first item")
        else:
            print(f"   SKIPPED: No items available from get_matching_board_items")
    except EIOSClientError as e:
        print(f"   FAILED: {e}\n")

    # Test get_filter_values_lookup endpoint (authenticated)
    print("8. Testing get_filter_values_lookup (authenticated)...")
    try:
        result = client.get_filter_values_lookup()
        with open("eios_api_filter_values.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"   OK: result saved to eios_api_filter_values.json\n")
    except EIOSClientError as e:
        print(f"   FAILED: {e}\n")

    # Test get_filter_values_lookup with filter_types parameter
    print("9. Testing get_filter_values_lookup with filter_types='sourceSubjects'...")
    try:
        result = client.get_filter_values_lookup(filter_types="sourceSubjects")
        print(f"   OK: result={result}\n")
    except EIOSClientError as e:
        print(f"   FAILED: {e}\n")

    # Test filter_items endpoint (authenticated)
    print("10. Testing filter_items (authenticated)...")
    try:
        # query = {
        #     "rules": [
        #         {
        #             "operator": "dateRange",
        #             "property": "processedOnDate",
        #             "value": {
        #                 "gte": adapt_date_format(startdate_ts_string), #"2009-01-01T00:00:00.000Z",
        #                 "lte": adapt_date_format(enddate_ts_string) #"2023-12-31T23:59:59.000Z"
        #             },
        #             "name": "filterDate"
        #         },
        #         {
        #             "name": "categories-1",
        #             "operator": "in",
        #             "property": "categories",
        #             "value": [
        #                 "cat:FrancisellaTularensis",
        #                 "cat:HantaFever",
        #                 "cat:Leptospirosis",
        #                 "cat:Lymedisease",
        #                 "cat:Tick-borneEncephalitis"
        #             ]
        #         }
        #     ],
        #     "groups": [
        #         {
        #             "name": "includeSourcesGroup",
        #             "operator": "or",
        #             "rules": [
        #                 {
        #                     "operator": "in",
        #                     "property": "source.id",
        #                     "value": [
        #                         "GPHIN",
        #                         "gphin",
        #                         "promed-mail"
        #                     ],
        #                     "name": "includeSourcesRule"
        #                 }
        #             ],
        #             "groups": [
        #                 {
        #                     "name": "sourceFiltersGroup",
        #                     "operator": "and",
        #                     "rules": [],
        #                     "groups": []
        #                 }
        #             ]
        #         }
        #     ],
        #     "operator": "and",
        #     "start": 0,
        #     "limit": 10,
        #     "sorts": [
        #         {
        #             "property": "processedOnDate",
        #             "direction": "desc"
        #         }
        #     ],
        #     "boardId": "digital-health-Medical-Official-PH"
        # }
        #
        # test without specyfiyn boardID:
        # query = {
        #     "rules": [
        #         {
        #             "operator": "dateRange",
        #             "property": "processedOnDate",
        #             "value": {
        #                 "gte": adapt_date_format(startdate_ts_string),  #"2022-01-01T00:00:00.000Z",
        #                 "lte": adapt_date_format(enddate_ts_string)  #"2023-12-31T23:59:59.000Z"
        #             },
        #             "name": "filterDate"
        #         },
        #         {
        #             "name": "categories-1",
        #             "operator": "in",
        #             "property": "categories",
        #             "value": [
        #                 "cat:FrancisellaTularensis",
        #                 "cat:HantaFever",
        #                 "cat:Leptospirosis",
        #                 "cat:Lymedisease",
        #                 "cat:Tick-borneEncephalitis"
        #             ]
        #         }
        #     ],
        #     "groups": [
        #         {
        #             "name": "includeSourcesGroup",
        #             "operator": "or",
        #             "rules": [
        #                 {
        #                     "operator": "in",
        #                     "property": "source.id",
        #                     "value": [
        #                         "GPHIN",
        #                         "gphin",
        #                         "promed-mail"
        #                     ],
        #                     "name": "includeSourcesRule"
        #                 }
        #             ],
        #             "groups": [
        #                 {
        #                     "name": "sourceFiltersGroup",
        #                     "operator": "and",
        #                     "rules": [],
        #                     "groups": []
        #                 }
        #             ]
        #         }
        #     ],
        #     "operator": "and",
        #     "start": 0,
        #     "limit": 10,
        #     "sorts": [
        #         {
        #             "property": "processedOnDate",
        #             "direction": "desc"
        #         }
        #     ]
        # }
        #
        # promed only articles:
        query = {
            "rules": [
                {
                    "operator": "dateRange",
                    "property": "processedOnDate",
                    "value": {
                        #"gte": adapt_date_format(startdate_ts_string),  #"2023-10-01T00:00:00.000Z",
                        #"lte": adapt_date_format(enddate_ts_string)  #"2026-12-31T23:59:59.000Z"
                        "gte": adapt_date_format("2023-01-01T00:00"),  
                        "lte": adapt_date_format("2026-01-01T00:00")  
                    },
                    "name": "filterDate"
                }
            ],
            "groups": [
                {
                    "name": "includeSourcesGroup",
                    "operator": "or",
                    "rules": [
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
                    "groups": [
                        {
                            "name": "sourceFiltersGroup",
                            "operator": "and",
                            "rules": [],
                            "groups": []
                        }
                    ]
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
        #
        # #now only dons articles
        # query = {
        #     "rules": [
        #         {
        #             "operator": "dateRange",
        #             "property": "processedOnDate",
        #             "value": {
        #                 "gte": "2013-10-01T00:00:00.000Z",
        #                 "lte": "2026-12-31T23:59:59.000Z"
        #             },
        #             "name": "filterDate"
        #         }
        #     ],
        #     "groups": [
        #         {
        #             "name": "includeSourcesGroup",
        #             "operator": "or",
        #             "rules": [
        #                 {
        #                     "operator": "in",
        #                     "property": "source.id",
        #                     "value": [
        #                         "WHODON"
        #                     ],
        #                     "name": "includeSourcesRule"
        #                 }
        #             ],
        #             "groups": [
        #                 {
        #                     "name": "sourceFiltersGroup",
        #                     "operator": "and",
        #                     "rules": [],
        #                     "groups": []
        #                 }
        #             ]
        #         }
        #     ],
        #     "operator": "and",
        #     "start": 0,
        #     "limit": 100,
        #     "sorts": [
        #         {
        #             "property": "processedOnDate",
        #             "direction": "desc"
        #         }
        #     ]
        # }
        result = client.filter_items(query)
        #result = test_client.filter_items(query)
        print(f"   OK: result={result}\n")
    except EIOSClientError as e:
        print(f"   FAILED: {e}\n")

    
    print(f"=== Version '{api_version}' WORKS! ===\n")

if __name__ == "__main__":
    main()
