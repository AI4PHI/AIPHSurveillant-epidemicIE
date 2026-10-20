"""Simple EIOS API client for notebooks and scripts.

This client targets the EIOS API documented at:
https://eios.who.int/portal/api/swagger/index.html

Authentication uses Azure AD OAuth2 client credentials as described in the
OpenAPI description.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urljoin

import requests


class EIOSClientError(RuntimeError):
    """Raised when an EIOS request or authentication call fails."""


@dataclass
class TokenState:
    access_token: str
    expires_at: float
    token_type: str = "Bearer"


class EIOSClient:
    """Single-file EIOS API client.

    Parameters
    ----------
    tenant_id:
        Azure AD tenant id used by EIOS token acquisition.
    client_id:
        OAuth2 client id (consumer application id).
    client_secret:
        OAuth2 client secret.
    scope:
        EIOS API scope (the value documented as EIOS_CLIENT_ID_SCOPE).
    base_url:
        EIOS API base URL. Defaults to production.
    api_version:
        Path version used by endpoints containing ``/v{version}``. Default ``1.0``.
    timeout:
        Request timeout in seconds.
    session:
        Optional requests.Session instance.
    """

    DEFAULT_BASE_URL = "https://eios.who.int/portal/api"
    
    DEFAULT_TEST_URL = "https://eios-test.who.int/portal/api"
    
    DEFAULT_API_VERSION = "1.0"

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        scope: str,
        base_url: str = DEFAULT_BASE_URL,
        api_version: str = DEFAULT_API_VERSION,
        timeout: int = 600,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.timeout = timeout

        self.session = session or requests.Session()
        self._token_state: Optional[TokenState] = None

    @property
    def token_url(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"

    def _token_valid(self) -> bool:
        return bool(self._token_state and time.time() < self._token_state.expires_at)

    def acquire_token(self, force_refresh: bool = False) -> str:
        """Acquire (or refresh) the bearer token and return it."""
        if not force_refresh and self._token_valid():
            return self._token_state.access_token

        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.scope,
        }

        response = self.session.post(
            self.token_url,
            data=payload,
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            raise EIOSClientError(
                f"Token acquisition failed ({response.status_code}): {response.text}"
            )

        token_payload = response.json()
        access_token = token_payload.get("access_token")
        expires_in = int(token_payload.get("expires_in", 3600))
        token_type = token_payload.get("token_type", "Bearer")

        if not access_token:
            raise EIOSClientError("Token response did not contain 'access_token'.")

        self._token_state = TokenState(
            access_token=access_token,
            token_type=token_type,
            expires_at=time.time() + max(1, expires_in - 60),
        )
        return access_token

    def clear_token(self) -> None:
        """Clear cached access token."""
        self._token_state = None

    def set_access_token(
        self,
        access_token: str,
        expires_in: int = 3600,
        token_type: str = "Bearer",
    ) -> None:
        """Set a pre-obtained bearer token.

        Useful when a token is fetched outside this client (e.g. Insomnia/Postman).
        """
        self._token_state = TokenState(
            access_token=access_token,
            token_type=token_type,
            expires_at=time.time() + max(1, expires_in - 60),
        )

    def _resolve_path(self, path: str, path_params: Optional[Mapping[str, Any]]) -> str:
        path_value = path
        if "{version}" in path_value:
            path_value = path_value.replace("{version}", self.api_version)
        if path_params:
            for key, value in path_params.items():
                path_value = path_value.replace(f"{{{key}}}", str(value))
        return path_value

    def _build_url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return urljoin(f"{self.base_url}/", path.lstrip("/"))

    def request(
        self,
        method: str,
        path: str,
        *,
        path_params: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
        json: Any = None,
        data: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        require_auth: bool = True,
        timeout: Optional[int] = None,
    ) -> Any:
        """Call an EIOS endpoint and return parsed response.

        The ``path`` parameter can be a full URL or an OpenAPI-like path template,
        for example ``/api/v{version}/UserProfiles/me``.
        """
        resolved_path = self._resolve_path(path, path_params)
        url = self._build_url(resolved_path)

        request_headers: Dict[str, str] = {}
        if headers:
            request_headers.update(headers)

        if require_auth:
            token = self.acquire_token()
            request_headers.setdefault("Authorization", f"Bearer {token}")

        response = self.session.request(
            method=method.upper(),
            url=url,
            params=params,
            json=json,
            data=data,
            headers=request_headers,
            timeout=timeout or self.timeout,
        )

        if response.status_code >= 400:
            raise EIOSClientError(
                f"EIOS request failed ({response.status_code}) {method.upper()} {url}: {response.text}"
            )

        content_type = response.headers.get("Content-Type", "").lower()
        if "application/json" in content_type or "text/json" in content_type:
            return response.json()
        if not response.content:
            return None
        return response.text

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Any:
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> Any:
        return self.request("PUT", path, **kwargs)

    def patch(self, path: str, **kwargs: Any) -> Any:
        return self.request("PATCH", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Any:
        return self.request("DELETE", path, **kwargs)

    def hello_boards(self) -> Any:
        return self.get("/api/v{version}/Boards/hello", require_auth=False)

    def hello_items(self) -> Any:
        return self.get("/api/v{version}/Items/hello", require_auth=False)

    def filter_items(
        self,
        query: Mapping[str, Any]
    ) -> Any:
        """Filter items using an EIOS query payload.

        Parameters
        ----------
        query:
            JSON-like dictionary matching the body expected by
            ``POST /api/v{version}/Items/filter``.
        """

        return self.post(
            "/api/v{version}/Items/filter",
            json=query,
            require_auth=True,
        )
        """
        return self.post(
            "/api/v{version}/Items/filter",
            json=query,
            headers=headers,
            require_auth=require_auth,
        )
        """

    def get_pinned_to_board(
        self,
        boardId: str,
        start: int = 0,
        limit: int = 100
    ) -> Any:
        return self.get(
            "/api/v{version}/Items/pinned-to-boards?" +
            "boardIds={boardIds}&start={start}&limit={limit}",
            require_auth=True,
            path_params={
                'boardIds': boardId,
                'start': start,
                'limit': limit
            }
        )

    def get_pinned_to_boards(
        self,
        boardIds: list[str],
        start: int = 0,
        limit: int = 100
    ) -> Any:

        delimiter = ','
        boardIds_str = delimiter.join(boardIds)

        return self.get(
            "/api/v{version}/Items/pinned-to-boards?" +
            "boardIds={boardIds}&start={start}&limit={limit}",
            require_auth=True,
            path_params={
                'boardIds': boardIds_str,
                'start': start,
                'limit': limit
            }
        )

    def get_posts_in_board(
        self,
        boardId: str,
        start: int = 0,
        limit: int = 10
    ) -> Any:
        query = {
          "rules": [
            {
              "operator": "in",
              "property": "referencedBoardId",
              "value": [
                boardId
              ]
            },
            {
              "operator": "in",
              "property": "status",
              "value": [
                "Published"
              ]
            }
          ],
          "groups": [],
          "operator": "and",
          "start": start,
          "limit": limit,
          "sorts": [
            {
              "property": "lastModifiedOnDate",
              "direction": "DESC"
            }
          ]
        }

        return self.post(
            "/api/v{version}/Posts",
            json=query,
            require_auth=True,
                )

    def hello_posts(self) -> Any:
        return self.get("/api/v{version}/Posts/hello", require_auth=False)

    def filter_items(
        self,
        query: Mapping[str, Any]
    ) -> Any:
        """Filter items using a tree-structured filter against the EIOS items index.

        Parameters
        ----------
        query:
            A Group describing how to combine rules and nested groups.
            Supports filtering, paging, sorting, and aggregations.

        Returns
        -------
        dict
            Paged list of items matching the filter.
        """
        return self.post(
            "/api/v{version}/Items/filter",
            json=query,
            require_auth=True,
        )

    def get_matching_board_items(
        self,
        boardId: str,
        start: Optional[int] = None,
        limit: Optional[int] = None,
        time_since: Optional[str] = None,
        time_until: Optional[str] = None,
    ) -> Any:
        """Obtain a paginated list of articles that match board filter definition or are pinned to the board.

        Parameters
        ----------
        boardId:
            The board ID.
        start:
            Pagination start index (default 0 server-side).
        limit:
            Number of items to return (default 100, max 100 server-side).
        time_since:
            Oldest fetch-date to consider (ISO 8601 datetime or ES date-math
            expression, e.g. ``"now-28d/d"``).  Optional.
        time_until:
            Newest fetch-date to consider (ISO 8601 datetime or ES date-math
            expression).  Optional.
        """
        params: dict[str, Any] = {}
        if start is not None:
            params['start'] = start
        if limit is not None:
            params['limit'] = limit
        if time_since is not None:
            params['timeSince'] = time_since
        if time_until is not None:
            params['timeUntil'] = time_until

        return self.get(
            "/api/v{version}/Items/matching-board/{boardId}",
            path_params={'boardId': boardId},
            params=params,
            require_auth=True,
                )

    def get_item_details(self, itemId: str) -> Any:
        """Retrieve the full details of an article by its ID.

        Parameters
        ----------
        itemId:
            Processing-chain ID of the item (e.g.
            ``"newsmedical-734672af836513db74636d1ee1e1f51c"``).

        Returns
        -------
        dict
            Full article payload including ``fullText``, ``translatedText``,
            ``abstractiveSummary``, etc.
        """
        return self.get(
            "/api/v{version}/Items/{itemId}",
            path_params={'itemId': itemId},
            require_auth=True,
                )

    def get_filter_values_lookup(self, filter_types: Optional[str] = None) -> Any:
        """Lookup filter values - retrieves available metadata for filter values.

        Parameters
        ----------
        filter_types:
            Comma-separated list of filter types. Optional.

        Returns
        -------
        dict
            Available metadata for the specified filter types.
        """
        params = {}
        if filter_types is not None:
            params['filterTypes'] = filter_types
        return self.get(
            "/api/v{version}/Items/filter-values-lookup",
            params=params if params else None,
            require_auth=True,
        )


    def hello_user_profiles(self) -> Any:
        return self.get("/api/v{version}/UserProfiles/hello", require_auth=False)

    def get_current_user_profile(self) -> Any:
        return self.get("/api/v{version}/UserProfiles/me", require_auth=True)

    def accept_terms_of_use(self) -> Any:
        return self.put(
            "/api/v{version}/UserProfiles/me",
            require_auth=True,
            json={"termsOfUseAccepted": True}
        )

    def get_all_taxonomy_nodes(
        self
    ) -> Any:
        return self.get(
            "/api/v{version}/taxonomy/all-nodes",
            require_auth=True,
        )

    def get_taxonomy_node_info(
        self,
        node_key: str
    ) -> Any:
        return self.get(
            "/api/v{version}/taxonomy/{node_key}",
            require_auth=True,
            path_params={
                'node_key': node_key
            }
        )

    def get_descendant_categories(
        self,
        group_key: str
    ) -> Any:
        return self.get(
            "/api/v{version}/taxonomy/GetDescendantCategories"
            + "?groupid={group_key}",
            require_auth=True,
            path_params={
                'group_key': group_key
            }
        )

    def get_timeseries_aggregation(
         self,
         categoryOrFolder: str,
         startDate: str,
         endDate: str,
         excludeDuplicates: bool = True,
         excludedCegoryOrFolder: str = None
    ) -> Any:
        query = {
            "operator": "and",
            "rules": [
                {
                    "name": "categories-1",
                    "operator": "in",
                    "property": "categories",
                    "value": [
                        categoryOrFolder
                    ]
                },
                {
                    "name": "filterDate",
                    "operator": "dateRange",
                    "property": "pubDate",
                    "value": {
                        "gte": startDate,
                        "lt": endDate
                    },
                },
            ],
            "groups": [],
            "start": 0,
            "limit": 0,
            "sorts": [],
            "aggregations": [
                {
                    "name": "timeseries",
                    "type": "date",
                    "Property": "pubDate",
                    "interval": "day",
                    "keyFormat": "yyyy-MM-dd",
                    "computeMovingAverage": True,
                }
            ]
        }
        if excludeDuplicates:
            query['groups'].append({
                 "name": "includeSourcesGroup",
                 "operator": "or",
                 "rules": [],
                 "groups": [
                     {
                      "name": "sourceFiltersGroup",
                      "operator": "and",
                      "rules": [
                        {
                          "name": "duplicatesSourceFiltersRule",
                          "operator": "equals",
                          "property": "isDuplicate",
                          "value": False
                        }
                      ]
                     }
                  ]
             })
        
        if excludedCegoryOrFolder is not None:
            query['groups'].append({
              "name": "categories-not",
              "operator": "not",
              "rules": [
                {
                  "name": "categories-2",
                  "operator": "in",
                  "property": "categories",
                  "value": [
                    excludedCegoryOrFolder
                  ]
                }
              ]
            })
        # print(query)
        return self.filter_items(query)


if __name__ == "__main__":
    print("EIOSClient single-file module. Import EIOSClient in your script or notebook.")
    print("Quick example:")
    print("  from eios_client import EIOSClient")
    print("  client = EIOSClient(tenant_id='...', client_id='...', client_secret='...', scope='...')")
    print("  print(client.hello_user_profiles())")
    print("  print(client.get_current_user_profile())")