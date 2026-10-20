"""Publication of the shareable outputs of the pipeline on the JRC Research Data repository.

The repository (https://research-data.ec.europa.eu) is an InvenioRDM instance. A published record
cannot be modified in place: updating its files means publishing a NEW VERSION of it, i.e.

  1. POST   /api/records/{recordId}/versions                   -> draft of the new version (new id)
  2. PUT    /api/records/{draftId}/draft                       -> its metadata (InvenioRDM clears the
                                                                  publication date of a new version)
  3. per file:
     POST   /api/records/{draftId}/draft/files                 -> declare the file
     PUT    /api/records/{draftId}/draft/files/{key}/content   -> upload its bytes
     POST   /api/records/{draftId}/draft/files/{key}/commit    -> finalise it
  4. POST   /api/records/{draftId}/draft/actions/publish       -> publish the new version

The record id given to publish_new_version() may be the id of ANY version of the record: the new
version is always created from the latest one. Nothing is published when the local files are
identical (same MD5) to the ones of the latest published version, so that a run that brought no new
data does not add an empty version. If anything goes wrong before the publication, the draft is
deleted again, so that the next run starts from a clean state.

Credentials. Two kinds of token are supported, read from gitignored *.key files of the project
directory and never printed or logged:
  - JRC-ResearchData-Token.key      a personal access token of the owner of the record (Applications
                                    -> Personal access tokens in the repository), sent as
                                    "Authorization: Bearer". This is what the write operations
                                    normally require, and it is used when the file is present.
  - JRC-ResearchData-ShareLink.key  the token of a secret share link of the record, sent as the
                                    "token" query parameter. Enough to read a restricted record;
                                    whether it allows creating and publishing versions depends on
                                    the permission of the link (view / preview / edit).
"""
import datetime
import hashlib
import logging
import os

import requests

RESEARCH_DATA_API = "https://research-data.ec.europa.eu/api"
RESEARCH_DATA_RECORD_ID = "29t6t-1pj46"   # "AI Public Health Surveillant"
PERSONAL_TOKEN_FILE = "JRC-ResearchData-Token.key"
SHARE_LINK_TOKEN_FILE = "JRC-ResearchData-ShareLink.key"

REQUEST_TIMEOUT = 60           # seconds, for the metadata calls
UPLOAD_TIMEOUT = 3 * 3600      # seconds, for the upload of one file (several hundred MB)

MIMETYPES = {".csv": "text/csv", ".ttl": "text/turtle", ".rdf": "application/rdf+xml"}


class ResearchDataError(Exception):
    """A call to the repository failed; the message never contains the token."""


def _read_token(directory, filename):
    path = os.path.join(directory, filename)
    if os.path.isfile(path):
        with open(path) as f:
            token = f.read().strip()
        return token or None
    return None


def make_session(directory=None):
    """A requests session authenticated with the personal token if available, else the share link.

    Returns (session, params, auth_kind), where params must be passed to every call (it carries the
    share-link token when that is the credential in use).
    """
    directory = directory or os.path.dirname(os.path.abspath(__file__))
    session = requests.Session()
    session.headers["Accept"] = "application/json"
    personal = _read_token(directory, PERSONAL_TOKEN_FILE)
    if personal:
        session.headers["Authorization"] = "Bearer " + personal
        return session, {}, "personal access token"
    share = _read_token(directory, SHARE_LINK_TOKEN_FILE)
    if share:
        return session, {"token": share}, "share link"
    return session, {}, "anonymous"


def _call(session, params, method, path, expected=(200, 201, 202, 204), **kwargs):
    """Perform one API call and return its parsed JSON (or None), raising ResearchDataError."""
    url = RESEARCH_DATA_API + path
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    try:
        response = session.request(method, url, params=params, **kwargs)
    except requests.RequestException as err:
        raise ResearchDataError(f"{method} {path}: {type(err).__name__}: {err}") from None
    if response.status_code not in expected:
        try:
            detail = response.json()
            detail = detail.get("message") or detail
            errors = response.json().get("errors")
            if errors:
                detail = f"{detail} {errors}"
        except ValueError:
            detail = response.text[:300]
        raise ResearchDataError(f"{method} {path}: HTTP {response.status_code}: {detail}")
    if response.status_code == 204 or not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def md5_of(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_published_files(session, params, record_id):
    """(id of the latest published version, {file key: md5}) of the record."""
    latest = _call(session, params, "GET", f"/records/{record_id}/versions/latest",
                   allow_redirects=True)
    latest_id = latest["id"]
    files = _call(session, params, "GET", f"/records/{latest_id}/files")
    checksums = {entry["key"]: str(entry.get("checksum", "")).replace("md5:", "")
                 for entry in (files or {}).get("entries", [])}
    return latest_id, checksums


def publish_new_version(file_paths, record_id=RESEARCH_DATA_RECORD_ID, version=None,
                        publication_date=None, directory=None, dry_run=False, log=print):
    """Publish `file_paths` as a new version of the record, replacing the files of the same name.

    version          : the version label of the new version (default: today, YYYY-MM-DD).
    publication_date : its publication date (default: today).
    dry_run          : only compare the local files with the latest published version and report
                       what would be done; nothing is written to the repository.
    Returns the id of the published version, or None when nothing had to be published.
    Raises ResearchDataError on failure (after having deleted the draft it created).
    """
    today = datetime.date.today().isoformat()
    version = version or today
    publication_date = publication_date or today
    for path in file_paths:
        if not os.path.isfile(path):
            raise ResearchDataError(f"local file not found: {path}")

    session, params, auth_kind = make_session(directory)
    log(f"[research-data] record {record_id}, authenticated with: {auth_kind}")

    latest_id, published = latest_published_files(session, params, record_id)
    local = {os.path.basename(p): md5_of(p) for p in file_paths}
    changed = [key for key in local if published.get(key) != local[key]]
    log(f"[research-data] latest published version {latest_id}: "
        f"{len(changed)} of {len(local)} file(s) differ from the local ones {changed}")
    if not changed:
        log("[research-data] nothing new to publish")
        return None
    if dry_run:
        log(f"[research-data] dry run: a new version '{version}' would be published with "
            f"{sorted(local)}")
        return None

    draft = _call(session, params, "POST", f"/records/{latest_id}/versions")
    draft_id = draft["id"]
    log(f"[research-data] draft of the new version: {draft_id}")
    try:
        # the metadata of the previous version is carried over; only the publication date (which
        # InvenioRDM clears on a new version) and the version label are set
        metadata = dict(draft.get("metadata") or {})
        metadata["publication_date"] = publication_date
        metadata["version"] = version
        body = {"metadata": metadata, "access": draft.get("access"), "files": {"enabled": True}}
        if draft.get("custom_fields"):
            body["custom_fields"] = draft["custom_fields"]
        _call(session, params, "PUT", f"/records/{draft_id}/draft", json=body)

        # files possibly carried over into the draft are removed first, then the new ones are uploaded
        existing = _call(session, params, "GET", f"/records/{draft_id}/draft/files") or {}
        for entry in existing.get("entries", []):
            if entry["key"] in local:
                _call(session, params, "DELETE", f"/records/{draft_id}/draft/files/{entry['key']}")
        _call(session, params, "POST", f"/records/{draft_id}/draft/files",
              json=[{"key": key} for key in local])
        for path in file_paths:
            key = os.path.basename(path)
            log(f"[research-data] uploading {key} ({os.path.getsize(path) / 1e6:.1f} MB)")
            with open(path, "rb") as content:
                _call(session, params, "PUT", f"/records/{draft_id}/draft/files/{key}/content",
                      data=content, timeout=UPLOAD_TIMEOUT,
                      headers={"Content-Type": "application/octet-stream"})
            committed = _call(session, params, "POST",
                              f"/records/{draft_id}/draft/files/{key}/commit")
            remote_md5 = str((committed or {}).get("checksum", "")).replace("md5:", "")
            if remote_md5 and remote_md5 != local[key]:
                raise ResearchDataError(f"checksum mismatch after the upload of {key}")

        published_record = _call(session, params, "POST",
                                 f"/records/{draft_id}/draft/actions/publish")
    except Exception:
        # leave no half-made draft behind: the next run starts again from the published version
        try:
            _call(session, params, "DELETE", f"/records/{draft_id}/draft")
            log(f"[research-data] draft {draft_id} deleted after the failure")
        except ResearchDataError as cleanup_err:
            log(f"[research-data] could not delete the draft {draft_id}: {cleanup_err}")
            logging.error(f"Research data: could not delete the draft {draft_id}: {cleanup_err}")
        raise
    log(f"[research-data] published version {published_record.get('id', draft_id)} "
        f"('{version}'): {published_record.get('links', {}).get('self_html', '')}")
    return published_record.get("id", draft_id)
