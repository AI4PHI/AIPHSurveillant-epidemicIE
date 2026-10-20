#!/usr/bin/env python
"""Semantic modelling and entity linking for the epidemiological knowledge graph (eKG).

Given the structured extractions produced by the fine-tuned model, this module:

  - builds the eKG for a day on top of the European Crisis Management Ontology (ECMO), starting
    from the ontology preamble (``epidemicIE_PREAMBLE.ttl``);
  - performs entity linking of places to GeoNames and of diseases to biomedical ontologies,
    either through a Virtuoso SPARQL endpoint (covering many ontologies at once) or through the
    NCBO BioPortal REST API;
  - converts the resulting Turtle graph to RDF/XML and (optionally) uploads it to a Virtuoso
    triple store via WebDAV.

The public functions used by the pipeline are ``writeOntoFile`` (build the day graph) and
``convert_turtle_to_rdf``. Credentials (GeoNames / BioPortal / Virtuoso) are read at run time from
``*.key`` files in the script directory and are never hardcoded; see ``keys/README.md``.
"""

# Import libraries
import sys
print(sys.executable)
print(sys.version)
print(sys.version_info)
print(sys.prefix)
print(sys.path)

import json
import sqlite3
import urllib.request
import time
import random
import datetime
import os, argparse
import getpass
import json
import pandas as pd
from pathlib import Path
import shutil
import csv
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import re
from itertools import combinations
from collections import Counter

import subprocess
import rdflib
from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from requests.auth import HTTPBasicAuth
import logging  #https://docs.python.org/2/howto/logging.html

from virtuosoQueryRest import sparqlQuery

#######################################################################################

POSSIBLE_KGchoices_List = sorted(["AI", "AIO", "AEO", "BFO", "BIM", "BCGO", "CL", "CHIRO", "CHEBI", "DCM", "FMA", "GO", "GENO",
             "GeoSPARQL", "HL7", "DOID", "HP", "HP_O", "IDO", "IAO", "ICD10", "LOINC", "MESH",
             "MONDO", "NCIT", "NCBITAXON", "NCBITaxon_", "NIFCELL", "NIFSTD", "GML", "OBCS", "OCHV", "OHPI",
             "OPB", "TRANS", "PLOSTHES", "RADLEX", "RO", "STY", "SO", "SNOMED", "STATO",
             "SYMP", "FoodOn", "UBERON", "ORDO", "HOOM", "VO", "OGMS", "EuroSciVoc", "ARO", "REACTO", "go-lego", "go-lego-reacto", "PR", "PSIMOD", "pathway_like_go_cams", "ECMO"])

ONLY_Ontologies_OnBIOPORTAL = sorted(["AI", "AIO", "AEO", "BCGO", "BFO", "BIM", "CHEBI", "CHIRO", "CL", "DCM", "DOID", "FMA", "FOODON", "GENO", "GML", "GO", "GEOSPARQL", "HL7", "HP", "HP_O", "IAO", "ICD10", "IDO", "LOINC", "MESH", "MONDO", "NCBITAXON", "NCIT", "NIFCELL", "NIFSTD", "OBCS", "OCHV", "OHPI", "OPB", "PLOSTHES", "RADLEX", "OBOREL", "SNOMEDCT", "SO", "STATO", "STY", "SYMP", "PTRANS", "UBERON", "ORDO", "HOOM", "VO", "OGMS", "ARO", "PR", "PSIMOD"])





## taken from https://gist.github.com/jonlazaro/6934282 but not working
# def insert_by_post(rdf, graph_name, endpoint_url, username, passwd):
#     g = rdflib.Graph()
#     g.parse(data=rdf, format='turtle')
#
#     triples = ''
#     for s, p, o in g.triples((None, None, None)):
#         triple = "%s %s %s . " % (s.n3(), p.n3(), o.n3())
#         triples += triple
#
#     query = 'INSERT IN GRAPH <%s> { %s }' % (graph_name.encode('utf-8'), triples.encode('utf-8'))
#
#     r = requests.post(endpoint_url, data=query, headers={'Content-type': 'application/sparql-query'}, auth=HTTPBasicAuth(username, passwd))
#
#     if r.status_code != 201:
#         raise Exception("Update failed")


## taken from https://gist.github.com/jonlazaro/6934282 but not working
#def insert_by_put(rdf, endpoint_url, username, passwd):
#    r = requests.put(endpoint_url,data=rdf.encode('utf-8'),auth=HTTPBasicAuth(username, passwd),headers={'Content-Type': 'text/rdf+ttl'},)
#    print(r)

#Function: HTTP GET request and parse JSON
def getResponse(url):
    operUrl = urllib.request.urlopen(url)
    if(operUrl.getcode()==200):
        data = operUrl.read()
        jsonData = json.loads(data)
    else:
        print("Error receiving data", operUrl.getcode())
    return jsonData


################################################################################
#  HTTP access to the entity linking APIs
################################################################################

class ApiRequestError(Exception):
    """The linking API could not be reached, or did not answer with usable JSON.

    It means "unknown", NOT "no match": the caller must never write the outcome of such a failure
    to the linking caches, otherwise a transient error (a throttled request, a restart of the
    endpoint) would keep the term unlinked in every run to come.
    """


API_REQUEST_TIMEOUT = 30       # seconds allowed to a single request
API_REQUEST_RETRIES = 1        # attempts after the first one, on a throttled/broken answer
API_REQUEST_RETRY_WAIT = 5     # seconds waited before retrying


def _api_url_redacted(url):
    """The url without its query string: it carries the API keys, which must never be printed."""
    return str(url).split("?")[0]


def _api_get_json(url, DEBUG=False, timeout=API_REQUEST_TIMEOUT, retries=API_REQUEST_RETRIES,
                  retry_wait=API_REQUEST_RETRY_WAIT):
    """GET `url` and return its parsed JSON, raising ApiRequestError when it cannot be obtained.

    Every request is given a timeout, so that a hanging endpoint cannot block the pipeline for ever,
    and an answer that is only temporarily unusable (429 Too Many Requests, a 5xx, or a body that
    is not JSON at all, which is what a throttling page looks like) is retried before giving up.
    """
    last_err = ""

    for attempt in range(retries + 1):

        if attempt > 0:
            time.sleep(retry_wait)
            if DEBUG:
                print("   retrying " + _api_url_redacted(url) + " (" + str(last_err) + ")")

        try:
            response = requests.get(url, timeout=timeout)
        except Exception as err:
            last_err = f"{type(err).__name__}: {err}"
            continue

        if response.status_code == 429 or response.status_code >= 500:
            # throttled or temporarily broken: worth another attempt
            last_err = f"HTTP {response.status_code}"
            continue

        if response.status_code != 200:
            # a client error (bad key, bad request): retrying would not change anything
            raise ApiRequestError(f"HTTP {response.status_code} from {_api_url_redacted(url)}")

        try:
            return response.json()
        except Exception as err:
            # a 200 that does not carry JSON: typically a throttling or maintenance page
            last_err = f"answer is not JSON ({type(err).__name__}: {err})"
            continue

    raise ApiRequestError(str(last_err) + " from " + _api_url_redacted(url) + " after " +
                          str(retries + 1) + " attempt(s)")





################################################################################
#  eKG Turtle serialisation
#
#  One row of OutputAnnotatedTexts-FT-LLM-IterUpdate.csv describes ONE outbreak extracted
#  from ONE EIOS article: the 23 fine-grained fields of the fine-tuned extraction model
#  (see FT_EXTRACTION_FIELDS in llm_extraction_AutomatedDynamicPipeline.py) plus the columns
#  describing the article itself. The complete mapping of those fields onto the ECMO
#  ontology network is documented in the skos:editorialNote of epidemicIE_PREAMBLE.ttl.
################################################################################

EKG_DATASET_NS = "http://data.jrc.ec.europa.eu/dataset/1ff56505-cd10-4cbf-9bd3-fdb500fe43db/"
EKG_EXTRACTION_MODEL = "EpiQwen3.5-9B"
EU_COUNTRY_AUTHORITY_NS = "http://publications.europa.eu/resource/authority/country/"

# columns holding a disease name (linked to the biomedical KGs) and a place name (linked to GeoNames)
EKG_DISEASE_COLUMNS = ["DiseaseLevel1", "DiseaseLevel2"]
EKG_PLACE_COLUMNS = ["Country", "OutbreakEpicenter"]


def _ekg_escape(value):
    """Escape a value so that it can be written inside a Turtle short string literal."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("\"", "\\\"")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def _ekg_slug(value):
    """Build the local, IRI-safe part of a node name out of a free text value."""
    if value is None:
        return ""
    slug = re.sub(r"[^\w]+", "-", str(value).strip(), flags=re.UNICODE)
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-_")


def _ekg_value(row, column):
    """Return the cleaned string content of a cell, or "" when the cell is empty or NaN."""
    if column not in row.index:
        return ""
    value = row[column]
    try:
        if value is None or pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in ("", "nan", "none", "<na>", "null"):
        return ""
    return text


def _ekg_int(row, column):
    """Return a non negative integer for a count/date-part cell, or None when not available."""
    text = _ekg_value(row, column)
    if not text:
        return None
    text = text.replace(",", "").replace(" ", "").replace("~", "").replace("+", "")
    try:
        number = int(float(text))
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _ekg_partial_date(row, prefix):
    """Assemble the {prefix}Year / {prefix}Month / {prefix}Day columns into a date.

    Returns the pair (turtle literal, human readable string). The extraction very often
    yields incomplete dates, so the literal is typed xsd:date, xsd:gYearMonth or xsd:gYear,
    following the ekg:PartialDate datatype declared in the preamble. When no usable year is
    available the pair ("", "") is returned.
    """
    year = _ekg_int(row, prefix + "Year")
    if year is None or not (1500 <= year <= 2200):
        return "", ""

    month = _ekg_int(row, prefix + "Month")
    if month is not None and not (1 <= month <= 12):
        month = None
    day = _ekg_int(row, prefix + "Day")
    if day is not None and not (1 <= day <= 31):
        day = None

    if month is None:
        return f"\"{year:04d}\"^^xsd:gYear", f"{year:04d}"
    if day is not None:
        try:
            datetime.date(year, month, day)
            return (f"\"{year:04d}-{month:02d}-{day:02d}\"^^xsd:date",
                    f"{year:04d}-{month:02d}-{day:02d}")
        except ValueError:
            pass  # e.g. 31 February: keep the year and the month only
    return f"\"{year:04d}-{month:02d}\"^^xsd:gYearMonth", f"{year:04d}-{month:02d}"


def _ekg_iso_date(row, column, DEBUG=False):
    """Normalise a full date cell (the EIOS dates) to the YYYY-MM-DD string, or "" if unusable."""
    text = _ekg_value(row, column)
    if not text:
        return ""
    try:
        parsed = pd.to_datetime(text, errors='coerce', utc=True)
    except Exception as e:
        if DEBUG:
            print(f"An unexpected error occurred parsing {column}='{text}': {e}")
        return ""
    if pd.isna(parsed):
        return ""
    return parsed.strftime('%Y-%m-%d')


def _ekg_source_of(fileid):
    """Return the (observation source individual, provenance sentence) of an article."""
    lowered = str(fileid).lower()
    if "promed" in lowered:
        return "ekg:ProMEDSource", "Sourced from ProMED"
    if "who_dons" in lowered or "whodon" in lowered:
        return "ekg:WHODONSource", "Sourced from WHO Disease Outbreak News (DONs)"
    if "medicalofficial" in lowered:
        return "ekg:EIOSMedicalOfficialSource", "Sourced from the WHO EIOS Medical and Official sources"
    return "ekg:EIOSSource", "Sourced from the WHO EIOS platform"


def _ekg_title_of(fileid):
    """Shorten a fileid into a readable label, as it was done for the previous eKG versions."""
    title = str(fileid)
    title = title.replace(".pickle", "")
    title = title.replace("MedicalOfficial/", "M-")
    title = title.replace("TOTAL_ProMED/", "P-")
    title = title.replace("who_dons/", "D-")
    title = title.replace("/full-article", "")
    title = title.replace("https://portal.who.int/eios#/items/", "")
    title = title.replace("“", "\"").replace("”", "\"")
    return _ekg_escape(title)


def _ekg_safe_iri(url):
    """Return `url` when it can be written as an angle bracket IRI, "" otherwise."""
    text = str(url or "").strip()
    if not (text.startswith("http://") or text.startswith("https://")):
        return ""
    # angle bracket IRIs cannot contain these characters
    if any(char in text for char in (' ', '<', '>', '"', '{', '}', '|', '\\', '^', '`')):
        return ""
    return text


def _ekg_url_of(fileid):
    """Return the source URL carried by a fileid, or "" when the fileid is an internal identifier."""
    text = str(fileid)
    for prefix in ("MedicalOfficial/", "TOTAL_ProMED/", "who_dons/"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.replace(".pickle", "").strip()
    return _ekg_safe_iri(text)


def _ekg_write_node(turtle_f, subject_iri, statements):
    """Write one Turtle node out of a list of "predicate object" strings."""
    if not statements:
        return
    turtle_f.write("<" + subject_iri + ">\n")
    turtle_f.write("            " + " ;\n            ".join(statements) + " .\n\n")


def _ekg_collect_terms(df_eKG_data, columns, is_geo):
    """Collect the distinct values of 'columns' into the frame expected by process_row4Linking."""
    values = []
    for column in columns:
        if column in df_eKG_data.columns:
            values.extend(df_eKG_data[column].dropna().astype(str).str.strip().tolist())
    values = sorted({value for value in values if value and value.lower() not in ("nan", "none", "null")})

    df_terms = pd.DataFrame({'word': values})
    df_terms['IsBio'] = None if is_geo else 1
    df_terms['IsGeo'] = 1 if is_geo else None
    df_terms.reset_index(drop=True, inplace=True)
    return df_terms


def _ekg_run_linking(df_terms, DEBUG, key_geonames, cache_map_geonames, cacheGeonames_filename,
                     key_virtuoso, cache_map_virtuoso, cacheVirtuoso_filename, key_bioportal,
                     USE_CACHE, KG_restriction, UseBioportalForLinking):
    """Entity-link every term of df_terms; returns (df_terms, geonames cache, virtuoso cache)."""
    if df_terms.empty:
        return df_terms, cache_map_geonames, cache_map_virtuoso

    result = df_terms.apply(
        lambda row: process_row4Linking(row, row.name, DEBUG, key_geonames, cache_map_geonames,
                                        cacheGeonames_filename, key_virtuoso, cache_map_virtuoso,
                                        cacheVirtuoso_filename, key_bioportal, USE_CACHE,
                                        KG_restriction=KG_restriction,
                                        UseBioportalForLinking=UseBioportalForLinking), axis=1)

    try:
        df_terms['namedEntity'] = result.str[0]
        df_terms['exactURIs'] = result.str[1]
        df_terms['relatedURIs'] = result.str[2]
        cache_map_geonames = result.str[3].iloc[-1]
        cache_map_virtuoso = result.str[4].iloc[-1]
    except Exception as e:
        if DEBUG:
            print(f"Entity linking did not return any result, going on without links: {e}")
        df_terms['namedEntity'] = ""
        df_terms['exactURIs'] = [[] for _ in range(len(df_terms))]
        df_terms['relatedURIs'] = [[] for _ in range(len(df_terms))]

    def fill_exacturis(row):
        # with no exact match, the named entity itself (e.g. the GeoNames URI) is the only link
        if (not isinstance(row['exactURIs'], list) or not row['exactURIs']) \
                and pd.notnull(row['namedEntity']) and row['namedEntity']:
            return [row['namedEntity']]
        return row['exactURIs']

    df_terms['exactURIs'] = df_terms.apply(fill_exacturis, axis=1)
    return df_terms, cache_map_geonames, cache_map_virtuoso


def _ekg_lookup_links(df_terms, term):
    """Return the (namedEntity, exactURIs, relatedURIs) of an entity-linked term."""
    if df_terms is None or df_terms.empty or not term or 'namedEntity' not in df_terms.columns:
        return "", [], []
    matches = df_terms.loc[df_terms['word'] == term]
    if matches.empty:
        return "", [], []
    named = matches['namedEntity'].iloc[0]
    exact = matches['exactURIs'].iloc[0]
    related = matches['relatedURIs'].iloc[0]
    return (named if isinstance(named, str) else ""), \
           (exact if isinstance(exact, list) else []), \
           (related if isinstance(related, list) else [])


def _ekg_carries_outbreak(df_eKG_data):
    """Row mask of the extractions that carry an outbreak, i.e. of the rows that become a record.

    A row without any disease and without any country carries no outbreak at all: the fine-tuned
    model returned nothing for that article (see run_finetuned_epidemicExtractor).
    """
    if 'DiseaseLevel1' in df_eKG_data.columns and 'Country' in df_eKG_data.columns:
        return ~(df_eKG_data['DiseaseLevel1'].isna() & df_eKG_data['Country'].isna())
    return pd.Series(True, index=df_eKG_data.index)


def record_ids_of_extractions(df_eKG_data, record_id_tag):
    """Return the record identifier of every row of a frame of extractions, "" when it has none.

    They are the identifiers writeOntoFile() gives to the outbreak records it writes out of the same
    frame, i.e. "record<tag>_<position among the rows that carry an outbreak>", published as the
    rdfs:label of the ekg:OutbreakEvent. The rows that carry no outbreak have no record at all, and
    therefore no identifier: they are skipped by writeOntoFile() before the records are numbered,
    which is why the position is not the row number of the frame.
    """
    carries = _ekg_carries_outbreak(df_eKG_data).to_numpy()
    positions = carries.cumsum() - 1
    return pd.Series([f"record{record_id_tag}_{position}" if carry else ""
                      for carry, position in zip(carries, positions)], index=df_eKG_data.index)


def writeOntoFile(DEBUG, eKG_ontologyPREAMBLE_file_location, input_data_epidemicextractions, PerformEntityLinking, USE_CACHE, geonameskey_filename="GEONAMES-API.key", bioportalkey_filename="NCBO-BioPortal.key", virtuosokey_filename="VIRTUOSO-dba.key", output_ttl_path=None, KG_restriction=None, record_id_tag=None):
    """Write the epidemic knowledge graph of `input_data_epidemicextractions` in Turtle.

    `record_id_tag` is the tag used to build the identifiers of the outbreak records
    ("record<tag>_<index>"). The caller passes the "YYYYMMDD" day the extractions belong to, so
    that the records of two different days do not collide when several days are processed by the
    same run and merged in the overall knowledge graph. It defaults to the current day.
    """

    eKG_ontology_file_location=""

    try:

        df_eKG_data = pd.read_csv(input_data_epidemicextractions, sep=",", header=0, encoding='utf-8')

        # the full text of the source article is never written to the ontology: it cannot be
        # redistributed (copyright of the MedicalOfficial sources)
        for column in ['texts', 'json_original']:
            if column in df_eKG_data.columns:
                df_eKG_data = df_eKG_data.drop(columns=[column])

        # the rows that carry no outbreak are dropped here, before the records are numbered below:
        # record_ids_of_extractions() replays exactly this, so that the reduced extractions can carry
        # the identifier of the record of every row
        df_eKG_data = df_eKG_data[_ekg_carries_outbreak(df_eKG_data)]

        df_eKG_data.reset_index(drop=True, inplace=True)

        if not df_eKG_data.empty:

            df_DiseasesAnnotated = pd.DataFrame()
            df_PlacesAnnotated = pd.DataFrame()

            if PerformEntityLinking:  # Here start with the linking:

                # Determine whether to use BioPortal only (faster) or Virtuoso SPARQL
                # If all selected KGs are available on BioPortal, use BioPortal API directly
                # Otherwise use Virtuoso SPARQL endpoint (slower but covers all KGs)
                if KG_restriction is None:
                    KG_restriction = POSSIBLE_KGchoices_List

                UseBioportalForLinking = True
                if KG_restriction:
                    for choice in KG_restriction:
                        normalized = choice
                        if choice == "SNOMED":
                            normalized = "SNOMEDCT"
                        elif choice == "RO":
                            normalized = "OBOREL"
                        elif choice == "TRANS":
                            normalized = "PTRANS"
                        elif choice == "FoodOn":
                            normalized = "FOODON"
                        elif choice == "GeoSPARQL":
                            normalized = "GEOSPARQL"
                        elif choice == "NCBITaxon_":
                            normalized = "NCBITAXON"
                        if normalized not in ONLY_Ontologies_OnBIOPORTAL:
                            UseBioportalForLinking = False
                            break

                if DEBUG:
                    print(f"Entity Linking - KG_restriction: {KG_restriction}")
                    print(f"Entity Linking - UseBioportalForLinking: {UseBioportalForLinking}")

                cacheGeonames_filename = "CACHE_geonames.json"
                cache_map_geonames = None
                if USE_CACHE:
                    if os.path.exists(cacheGeonames_filename):
                        with open(cacheGeonames_filename) as f:
                            cache_map_geonames = json.load(f)
                    else:
                        cache_map_geonames = {}

                key_geonames = ""
                if geonameskey_filename and os.path.exists(geonameskey_filename):
                    fkeyname = geonameskey_filename
                    with open(fkeyname) as f:
                        key_geonames = f.read().strip()
                else:
                    key_geonames = os.environ['key_geonames']

                cacheVirtuoso_filename = "CACHE_virtuoso.json"
                cache_map_virtuoso = None
                if USE_CACHE:
                    if os.path.exists(cacheVirtuoso_filename):
                        with open(cacheVirtuoso_filename) as f:
                            cache_map_virtuoso = json.load(f)
                    else:
                        cache_map_virtuoso = {}

                key_virtuoso = ""
                virtuosokey_filename = "VIRTUOSO-dba.key"
                if os.path.exists(virtuosokey_filename):
                    with open(virtuosokey_filename) as f:
                        key_virtuoso = f.read().strip()

                key_bioportal = ""
                if bioportalkey_filename and os.path.exists(bioportalkey_filename):
                    fkeyname = bioportalkey_filename
                    with open(fkeyname) as f:
                        key_bioportal = f.read().strip()
                else:
                    key_bioportal = os.environ.get('key_bioportal', '')

                # first the places: the country and the epicenter of every outbreak, linked to GeoNames
                df_PlacesAnnotated = _ekg_collect_terms(df_eKG_data, EKG_PLACE_COLUMNS, is_geo=True)
                df_PlacesAnnotated, cache_map_geonames, cache_map_virtuoso = _ekg_run_linking(
                    df_PlacesAnnotated, DEBUG, key_geonames, cache_map_geonames, cacheGeonames_filename,
                    key_virtuoso, cache_map_virtuoso, cacheVirtuoso_filename, key_bioportal,
                    USE_CACHE, KG_restriction, UseBioportalForLinking)

                # then the diseases of level 1 and 2, linked to the biomedical ontologies
                df_DiseasesAnnotated = _ekg_collect_terms(df_eKG_data, EKG_DISEASE_COLUMNS, is_geo=False)
                df_DiseasesAnnotated, cache_map_geonames, cache_map_virtuoso = _ekg_run_linking(
                    df_DiseasesAnnotated, DEBUG, key_geonames, cache_map_geonames, cacheGeonames_filename,
                    key_virtuoso, cache_map_virtuoso, cacheVirtuoso_filename, key_bioportal,
                    USE_CACHE, KG_restriction, UseBioportalForLinking)

                if USE_CACHE:
                    if cache_map_geonames is not None:
                        with open(cacheGeonames_filename, "w") as f:
                            json.dump(cache_map_geonames, f)
                    if cache_map_virtuoso is not None:
                        with open(cacheVirtuoso_filename, "w") as f:
                            json.dump(cache_map_virtuoso, f)
                ###

            if output_ttl_path:
                eKG_ontology_file_location = output_ttl_path
            else:
                current_date = datetime.datetime.now()
                current_date_str = current_date.strftime("%Y%m%d")
                eKG_ontology_file_location = eKG_ontologyPREAMBLE_file_location.replace("PREAMBLE", current_date_str)
            try:
                # Copy the file from the source to the destination
                shutil.copy(eKG_ontologyPREAMBLE_file_location, eKG_ontology_file_location)
                if DEBUG:
                    print(f"The file has been successfully copied to {eKG_ontology_file_location}")
            except FileNotFoundError:
                print(f"Error: The file at {eKG_ontologyPREAMBLE_file_location} was not found.")
                eKG_ontology_file_location=""
                return eKG_ontology_file_location
            except PermissionError:
                print("Error: You do not have permission to read from the source or write to the destination.")
                eKG_ontology_file_location=""
                return eKG_ontology_file_location
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                eKG_ontology_file_location=""
                return eKG_ontology_file_location

            turtle_f = open(eKG_ontology_file_location, 'a', encoding='utf-8')

            # the records are tagged with the day the extractions belong to when the caller gives it
            # (see the docstring), and with the day of the run otherwise
            record_tag_str = str(record_id_tag).strip() if record_id_tag else datetime.datetime.now().strftime("%Y%m%d")

            # the disease, place and article nodes are shared by all the outbreaks referring to
            # them, hence they are serialised only once
            written_diseases = set()
            written_places = set()
            written_reports = set()

            turtle_f.write("\n")
            turtle_f.write("#################################################################\n")
            turtle_f.write("#    Outbreak events extracted from the EIOS articles\n")
            turtle_f.write("#################################################################\n")
            turtle_f.write("\n")

            for index, row in df_eKG_data.iterrows():

                id = f"record{record_tag_str}_{index}"

                record_iri = EKG_DATASET_NS + id

                ##### the source article the outbreak has been extracted from

                fileid = _ekg_value(row, 'fileid')
                source_individual, sourced = _ekg_source_of(fileid)
                title = _ekg_title_of(fileid) if fileid else id
                report_iri = EKG_DATASET_NS + "report-" + _ekg_slug(fileid) if fileid else ""

                date_published = _ekg_iso_date(row, 'date_pubDate_EIOS', DEBUG)
                date_processed = _ekg_iso_date(row, 'date_processedOnDate_EIOS', DEBUG)
                date_imputed = _ekg_iso_date(row, 'date_cases_IMPUTED', DEBUG)
                # the reporting date of the outbreak: the publication date of the article, or the
                # date imputed by the pipeline when the article carries no usable date
                date_reported = date_published or date_imputed

                ##### the diseases (level 1 and level 2) the outbreak is a manifestation of

                disease_iris = {}
                for column in EKG_DISEASE_COLUMNS:
                    disease = _ekg_value(row, column)
                    if disease and _ekg_slug(disease):
                        disease_iris[column] = EKG_DATASET_NS + "disease-" + _ekg_slug(disease)

                ##### the affected country and the epicenter of the outbreak

                country = _ekg_value(row, 'Country')
                country_iri = EKG_DATASET_NS + "country-" + _ekg_slug(country) if _ekg_slug(country) else ""
                country_geonames, _, _ = _ekg_lookup_links(df_PlacesAnnotated, country)

                iso_code = _ekg_value(row, 'ISO').upper()
                if not re.fullmatch(r"[A-Z]{3}", iso_code):
                    iso_code = ""

                epicenter = _ekg_value(row, 'OutbreakEpicenter')
                epicenter_iri = EKG_DATASET_NS + "place-" + _ekg_slug(epicenter) if _ekg_slug(epicenter) else ""
                epicenter_geonames, _, _ = _ekg_lookup_links(df_PlacesAnnotated, epicenter)

                ##### the reported human impact of the outbreak

                cases_total = _ekg_int(row, 'CasesTotal')
                cases_suspected = _ekg_int(row, 'CasesSuspected')
                cases_probable = _ekg_int(row, 'CasesProbable')
                cases_confirmed = _ekg_int(row, 'CasesConfirmed')
                deaths = _ekg_int(row, 'Deaths')
                has_impact = any(value is not None for value in
                                 (cases_total, cases_suspected, cases_probable, cases_confirmed, deaths))
                impact_iri = record_iri + "-impact"

                # the case fatality rate of this single outbreak, derived from the reported counts
                cfr_denominator = cases_total if cases_total else cases_confirmed
                cfr = None
                if deaths is not None and cfr_denominator and deaths <= cfr_denominator:
                    cfr = round(float(deaths) / float(cfr_denominator), 6)
                cfr_iri = record_iri + "-casefatalityrate"

                ##### the timeline of the outbreak

                date_start, date_start_str = _ekg_partial_date(row, 'OutbreakStart')
                date_detection, date_detection_str = _ekg_partial_date(row, 'OutbreakDetection')
                date_verification, date_verification_str = _ekg_partial_date(row, 'OutbreakVerification')
                date_end, date_end_str = _ekg_partial_date(row, 'OutbreakEnd')
                end_declaration = _ekg_value(row, 'OutbreakEnd')

                interval_iri = record_iri + "-period"
                detection_iri = record_iri + "-detection"
                validation_iri = record_iri + "-verification"

                ##### the natural language description of the outbreak

                disease_level1 = _ekg_value(row, 'DiseaseLevel1')
                disease_level2 = _ekg_value(row, 'DiseaseLevel2')
                desc = "Outbreak of "
                desc = desc + (_ekg_escape(disease_level1) if disease_level1 else "an unspecified disease")
                if disease_level2:
                    desc = desc + " (" + _ekg_escape(disease_level2) + ")"
                if country:
                    desc = desc + " occurred in " + _ekg_escape(country)
                    if epicenter:
                        desc = desc + ", with epicenter in " + _ekg_escape(epicenter) + ","
                elif epicenter:
                    desc = desc + " occurred in " + _ekg_escape(epicenter)
                if date_start_str:
                    # "on 1995-10-09" for a complete date, "in 1995-10" for an incomplete one
                    desc = desc + (" on " if len(date_start_str) == 10 else " in ") + date_start_str
                if date_reported:
                    desc = desc + " and reported on " + date_reported
                impact_parts = []
                if cases_total is not None:
                    impact_parts.append(f"{cases_total} total cases")
                if cases_suspected is not None:
                    impact_parts.append(f"{cases_suspected} suspected cases")
                if cases_probable is not None:
                    impact_parts.append(f"{cases_probable} probable cases")
                if cases_confirmed is not None:
                    impact_parts.append(f"{cases_confirmed} confirmed cases")
                if deaths is not None:
                    impact_parts.append(f"{deaths} deaths")
                if impact_parts:
                    desc = desc + ", with " + ", ".join(impact_parts[:-1])
                    if len(impact_parts) > 1:
                        desc = desc + " and " + impact_parts[-1]
                    else:
                        desc = desc + impact_parts[-1]
                if date_end_str:
                    desc = desc + ", declared over " + ("on " if len(date_end_str) == 10 else "in ") + date_end_str
                desc = desc + "."

                ##### the outbreak event itself

                # NOTE on the two naming statements: dcterms:identifier carries the identifier of the
                # source article the outbreak was extracted from (the shortened fileid, e.g.
                # "M-dpi.gov.gy-5314833c15209dbb8ea74a94511353ed"), while rdfs:label carries the
                # readable name of this very node, i.e. the record key "recordYYYYMMDD_<index>".
                # The same convention is used on the ekg:MediaReport node of the article below, and
                # the labels of the impact / CFR / interval / detection / verification nodes are all
                # built on top of this label ("Human impact of recordYYYYMMDD_<index>", ...).
                statements = ["a ekg:OutbreakEvent, hazard:BiologicalEvent, eios:PublicHealthEvent",
                              "dcterms:identifier \"" + title + "\"",
                              "dcterms:description \"" + desc + "\"",
                              "rdfs:label \"" + id + "\""]

                if country:
                    # kept for backward compatibility with the previous versions of the eKG
                    statements.append("locn:geographicName \"" + _ekg_escape(country) + "\"")
                if 'DiseaseLevel1' in disease_iris:
                    statements.append("ekg:hasDiseaseLevel1 <" + disease_iris['DiseaseLevel1'] + ">")
                if 'DiseaseLevel2' in disease_iris:
                    statements.append("ekg:hasDiseaseLevel2 <" + disease_iris['DiseaseLevel2'] + ">")
                if country_iri:
                    statements.append("ekg:hasAffectedCountry <" + country_iri + ">")
                if epicenter_iri:
                    statements.append("ekg:hasOutbreakEpicenter <" + epicenter_iri + ">")
                if country_geonames:
                    # kept for backward compatibility: the direct link to the GeoNames place
                    statements.append("foundation:hasLocation <" + country_geonames + ">")
                if date_reported:
                    statements.append("foundation:hasReportDate \"" + date_reported + "\"^^xsd:date")
                if date_imputed:
                    statements.append("ekg:hasImputedReportDate \"" + date_imputed + "\"^^xsd:date")
                if date_start:
                    statements.append("ekg:hasOutbreakStartDate " + date_start)
                if date_detection:
                    statements.append("ekg:hasOutbreakDetectionDate " + date_detection)
                if date_verification:
                    statements.append("ekg:hasOutbreakVerificationDate " + date_verification)
                if date_end:
                    statements.append("ekg:hasOutbreakEndDate " + date_end)
                if end_declaration:
                    statements.append("ekg:hasOutbreakEndDeclaration \"" + _ekg_escape(end_declaration) + "\"")
                if date_start or date_end:
                    statements.append("disaster:hasTemporalExtent <" + interval_iri + ">")
                if date_detection:
                    statements.append("ekg:isDetectedBy <" + detection_iri + ">")
                if date_verification:
                    statements.append("foundation:isConfirmedBy <" + validation_iri + ">")
                if has_impact:
                    statements.append("disaster:hasHumanImpact <" + impact_iri + ">")
                if cfr is not None:
                    statements.append("ekg:hasCaseFatalityRate <" + cfr_iri + ">")
                if report_iri:
                    statements.append("foundation:hasSignal <" + report_iri + ">")
                outbreak_index = _ekg_int(row, 'outbreak_index')
                if outbreak_index is not None:
                    statements.append("ekg:hasOutbreakIndex " + str(outbreak_index))
                n_outbreaks = _ekg_int(row, 'n_outbreaks')
                if n_outbreaks is not None:
                    statements.append("ekg:hasReportedOutbreakCount " + str(n_outbreaks))
                statements.append("ekg:hasExtractionModel \"" + EKG_EXTRACTION_MODEL + "\"")
                statements.append("rdfs:comment \"" + sourced + "\"")

                _ekg_write_node(turtle_f, record_iri, statements)

                ##### the human impact node

                if has_impact:
                    statements = ["a impact:HumanImpact",
                                  "rdfs:label \"Human impact of " + id + "\""]
                    if cases_total is not None:
                        statements.append("ekg:hasTotalCaseCount " + str(cases_total))
                    if cases_suspected is not None:
                        statements.append("ekg:hasSuspectedCaseCount " + str(cases_suspected))
                    if cases_probable is not None:
                        statements.append("ekg:hasProbableCaseCount " + str(cases_probable))
                    if cases_confirmed is not None:
                        statements.append("ekg:hasConfirmedCaseCount " + str(cases_confirmed))
                    if deaths is not None:
                        statements.append("ekg:hasDeathCount " + str(deaths))
                    statements.append("impact:hasImpactType impact:HumanImpactType")
                    statements.append("rdfs:comment \"Cases and deaths reported for the outbreak, extracted from the situational report.\"")
                    _ekg_write_node(turtle_f, impact_iri, statements)

                ##### the case fatality rate node

                if cfr is not None:
                    _ekg_write_node(turtle_f, cfr_iri, [
                        "a ph:CaseFatalityRate",
                        "ph:hasCaseFatalityRateValue \"" + f"{cfr:.6f}" + "\"^^xsd:decimal",
                        "rdfs:label \"Case fatality rate of " + id + "\"",
                        "rdfs:comment \"Case fatality rate of this single outbreak, computed as the "
                        + str(deaths) + " reported deaths over the " + str(cfr_denominator) + " reported cases.\""])

                ##### the temporal extent of the outbreak

                if date_start or date_end:
                    statements = ["a dul:TimeInterval",
                                  "rdfs:label \"Period of " + id + "\""]
                    if date_start:
                        statements.append("ekg:hasStartDate " + date_start)
                    if date_end:
                        statements.append("ekg:hasEndDate " + date_end)
                    statements.append("rdfs:comment \"Period over which the outbreak unfolded, from its start to its declared end.\"")
                    _ekg_write_node(turtle_f, interval_iri, statements)

                ##### the detection and the official verification of the outbreak

                if date_detection:
                    _ekg_write_node(turtle_f, detection_iri, [
                        "a ekg:OutbreakDetection",
                        "ekg:detectsEvent <" + record_iri + ">",
                        "ekg:hasDetectionDate " + date_detection,
                        "rdfs:label \"Detection of " + id + "\"",
                        "rdfs:comment \"Moment at which the outbreak was detected by the public health system.\""])

                if date_verification:
                    _ekg_write_node(turtle_f, validation_iri, [
                        "a foundation:EventValidation",
                        "foundation:confirmsEvent <" + record_iri + ">",
                        "ekg:hasValidationDate " + date_verification,
                        "rdfs:label \"Verification of " + id + "\"",
                        "rdfs:comment \"Official verification of the outbreak by the public health authorities.\""])

                ##### the diseases

                for column, disease_iri in disease_iris.items():
                    if disease_iri in written_diseases:
                        continue
                    written_diseases.add(disease_iri)

                    disease = _ekg_value(row, column)
                    disease_exact, disease_related = _ekg_lookup_links(df_DiseasesAnnotated, disease)[1:]

                    statements = ["a ph:InfectiousDiseaseOrSyndrome, hazard:Hazard",
                                  "rdfs:label \"" + _ekg_escape(disease) + "\"",
                                  "skos:prefLabel \"" + _ekg_escape(disease) + "\"@en"]
                    if column == "DiseaseLevel2" and 'DiseaseLevel1' in disease_iris:
                        # the level 2 disease is a specialisation of the level 1 one
                        statements.append("skos:broader <" + disease_iris['DiseaseLevel1'] + ">")
                    for uri in disease_exact:
                        if uri:
                            statements.append("owl:sameAs <" + str(uri) + ">")
                    for uri in disease_related:
                        if uri:
                            statements.append("skos:related <" + str(uri) + ">")
                    statements.append("rdfs:comment \"Name of hazard or infectious disease causing the event, extracted from the situational report.\"")
                    _ekg_write_node(turtle_f, disease_iri, statements)

                ##### the places

                if country_iri and country_iri not in written_places:
                    written_places.add(country_iri)
                    statements = ["a ekg:Country",
                                  "rdfs:label \"" + _ekg_escape(country) + "\"",
                                  "locn:geographicName \"" + _ekg_escape(country) + "\""]
                    if iso_code:
                        statements.append("ekg:hasISO3CountryCode \"" + iso_code + "\"")
                        statements.append("owl:sameAs <" + EU_COUNTRY_AUTHORITY_NS + iso_code + ">")
                    if country_geonames:
                        statements.append("owl:sameAs <" + country_geonames + ">")
                    statements.append("rdfs:comment \"Country where the hazard or infectious disease occurred, extracted from the situational report and linked to GeoNames.\"")
                    _ekg_write_node(turtle_f, country_iri, statements)

                if epicenter_iri and epicenter_iri not in written_places:
                    written_places.add(epicenter_iri)
                    statements = ["a ekg:SubnationalArea",
                                  "rdfs:label \"" + _ekg_escape(epicenter) + "\"",
                                  "locn:geographicName \"" + _ekg_escape(epicenter) + "\""]
                    if country_iri:
                        statements.append("dcterms:isPartOf <" + country_iri + ">")
                    if epicenter_geonames:
                        statements.append("owl:sameAs <" + epicenter_geonames + ">")
                    statements.append("rdfs:comment \"Epicenter of the outbreak inside the affected country, extracted from the situational report.\"")
                    _ekg_write_node(turtle_f, epicenter_iri, statements)

                ##### the source article

                if report_iri and report_iri not in written_reports:
                    written_reports.add(report_iri)
                    # the article is identified by the same string as the dcterms:identifier of the
                    # outbreak events extracted from it (the shortened fileid), so that the two can be
                    # joined on the identifier as well, and not only through foundation:hasSignal.
                    # The full fileid is not lost: its "M-"/"P-"/"D-" prefix maps back to the source
                    # corpus (also stated by foundation:isSignalFrom) and, when the fileid is a URL,
                    # the complete address is published as dcterms:source right below.
                    statements = ["a ekg:MediaReport",
                                  "dcterms:identifier \"" + title + "\"",
                                  "rdfs:label \"" + title + "\""]
                    url = _ekg_url_of(fileid)
                    if url:
                        statements.append("dcterms:source <" + url + ">")

                    # the media outlet the article comes from, as the EIOS api returns it in the
                    # "source" block of the item ("name", "id" and "url"): any of the three may be
                    # absent, and they are missing altogether for the articles harvested before
                    # these columns were added to the extractions (see FT_SOURCE_FIELDS).
                    source_name = _ekg_value(row, 'source_name_EIOS')
                    source_identifier = _ekg_value(row, 'source_id_EIOS')
                    source_homepage = _ekg_safe_iri(_ekg_value(row, 'source_url_EIOS'))
                    if source_name:
                        statements.append("ekg:hasSourceName \"" + _ekg_escape(source_name) + "\"")
                    if source_identifier:
                        statements.append("ekg:hasSourceIdentifier \"" + _ekg_escape(source_identifier) + "\"")
                    if source_homepage:
                        statements.append("ekg:hasSourceHomepage <" + source_homepage + ">")

                    if date_published:
                        statements.append("dcterms:issued \"" + date_published + "\"^^xsd:date")
                    if date_processed:
                        statements.append("ekg:hasProcessingDate \"" + date_processed + "\"^^xsd:date")
                    text_length = _ekg_int(row, 'texts_length')
                    if text_length is not None:
                        statements.append("ekg:hasTextLength " + str(text_length))
                    statements.append("foundation:isSignalFrom " + source_individual)

                    # the provenance sentence carries the outlet as well, so that it is readable in
                    # one go without following the three properties above
                    comment = sourced
                    outlet_details = []
                    if source_name:
                        outlet_details.append("outlet: " + _ekg_escape(source_name))
                    if source_identifier:
                        outlet_details.append("EIOS source id: " + _ekg_escape(source_identifier))
                    if source_homepage:
                        outlet_details.append("outlet homepage: " + _ekg_escape(source_homepage))
                    if outlet_details:
                        comment = sourced + " (" + ", ".join(outlet_details) + ")"
                    statements.append("rdfs:comment \"" + comment + "\"")
                    _ekg_write_node(turtle_f, report_iri, statements)

            turtle_f.close()

            return eKG_ontology_file_location

    except FileNotFoundError:
        if DEBUG:
            print(f"File {input_data_epidemicextractions} not found.")
    except pd.errors.EmptyDataError:
        if DEBUG:
            print(f"The file {input_data_epidemicextractions} is empty.")
    except pd.errors.ParserError:
        if DEBUG:
            print(f"Error parsing the file {input_data_epidemicextractions}.")
    except IOError:
        if DEBUG:
            print(f"Error writing to the file eKG_ontology_file.")
    except Exception as e:
        if DEBUG:
            print(f"An unexpected error occurred (writeOntoFile): {e}")

    return eKG_ontology_file_location


def WebDavUpload(DEBUG,eKG_ontology_file_location,virtuoso_connection_parameters,virtuoso_rdf_synk_location,virtuoso_credentials):

    isLoaded=False

    try:
        # Equivalent manual upload, for reference:
        #   curl -H "Content-Type: text/turtle" -T graph.ttl \
        #        <virtuoso_connection_parameters>/DAV/home/dba/rdfimport/ -u dba:<password>
        # --noproxy "": when a corporate proxy is configured in the environment but the Virtuoso
        # host is in no_proxy, the endpoint must still be reached through the proxy (same as
        # virtuoso_proxies() in virtuosoQueryRest.py).
        # --connect-timeout: fail fast if the endpoint is unreachable, without limiting the upload itself.
        status, output = subprocess.getstatusoutput(
            "curl --noproxy \"\" --connect-timeout 10 -H \"Content-Type: text/turtle\" -T " + str(eKG_ontology_file_location) + " " + str(
                virtuoso_connection_parameters) + str(virtuoso_rdf_synk_location) + " -u " + str(
                virtuoso_credentials) + " --verbose")

        max_retries = 20
        iteration = 0
        while (isLoaded==False) and (iteration < max_retries):

            time.sleep(5)  # (10)

            if (status == 0 and ("created" in output.lower()) == True):
                if DEBUG:
                    print(output)
                    print("    ...loaded " + eKG_ontology_file_location)
                isLoaded = True

            else:
                if DEBUG:
                    #print("File :" + eKG_ontology_file_location)
                    print("status " + str(status))
                    print("\n output \n " + output)
                    print("\n...wait...\n")
                iteration = iteration + 1
                if iteration >= max_retries:
                    print("\n!!! STOP !!! Ontology file not uploaded to Virtuoso! Something's wrong, Have a CHECK! Exiting now...")
                    sys.exit()

    except subprocess.CalledProcessError as cpe:
        print(f"Subprocess call failed: {cpe}")
        print("File :" + eKG_ontology_file_location)
        print("status " + str(cpe.returncode))
        print("\n output \n " + cpe.output.decode())
        sys.exit()
    except FileNotFoundError as fnf:
        print(f"File not found: {fnf}")
        sys.exit()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit()

    return isLoaded





def convert_turtle_to_rdf(DEBUG,input_turtle_file, output_rdf_file):
    # Create an RDF graph
    g = Graph()

    try:
        # Parse the Turtle file
        g.parse(input_turtle_file, format='turtle')

        # Serialize the graph to RDF format and write to the output file
        g.serialize(destination=output_rdf_file, format='xml')

        if DEBUG:
            print(f"Conversion successful: {input_turtle_file} -> {output_rdf_file}")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")


def geonames_api_call(word, DEBUG, key_geonames, cache_map_geonames):
    context = ""
    ALLURIScontext = []

    if cache_map_geonames is not None:
        if word in cache_map_geonames:
            if context in cache_map_geonames[word]:
                url_text = cache_map_geonames[word][context]
                if DEBUG:
                    print("RETRIEVED CACHED RESULT FOR:\n", word, " => ", url_text, "\n")
                return url_text, ALLURIScontext, cache_map_geonames

    url = f"http://api.geonames.org/search?name_equals={word}&maxRows=1&type=json&username={key_geonames}"

    try:
        data = _api_get_json(url, DEBUG)
        if data['geonames']:
            geonameId = data['geonames'][0]['geonameId']
            geonameUrl = "https://sws.geonames.org/" + str(geonameId) + "/"

            if cache_map_geonames is not None:
                if not word in cache_map_geonames:
                    cache_map_geonames[word] = {}
                cache_map_geonames[word][context] = geonameUrl

            return geonameUrl, ALLURIScontext, cache_map_geonames
        else:

            if cache_map_geonames is not None:
                if not word in cache_map_geonames:
                    cache_map_geonames[word] = {}
                cache_map_geonames[word][context] = None

            return None, None, cache_map_geonames

    except Exception as err:
        # nothing is written to the cache here: the call did not answer "no such place", it did not
        # answer at all, so the place has to be looked up again on the next run
        if DEBUG:
            print(f"GeoNames linking failed for '{word}': {err}")
        logging.error(f'ERROR on GeoNames linking for "{word}": {err}')

        return None, None, cache_map_geonames




def getUrlBioAndAllOtherBioConcepts(word, DEBUG, key_virtuoso, cache_map_virtuoso, key_bioportal, KG_restriction=None, USE_CACHE=True, UseBioportalForLinking=True):
    """Perform entity linking for biological concepts.

    Returns (entityBioeUrl, exactURIs, relatedURIs, cache_map_virtuoso):
      - entityBioeUrl: top exact match URI (or None)
      - exactURIs: list of URIs matched exactly by prefLabel/altLabel/synonym → owl:sameAs
      - relatedURIs: list of URIs matched fuzzily (not exact) → skos:related
      - cache_map_virtuoso: updated cache

    If UseBioportalForLinking=True: uses BioPortal API only (faster, when all KGs are on BioPortal).
    If UseBioportalForLinking=False: uses Virtuoso SPARQL endpoint; if no exact results, runs a
      fuzzy query using bif:contains (Virtuoso full-text index) for related matches.
    """
    entityBioeUrl = None
    exactURIs = []
    relatedURIs = []

    endpoint = 'https://api-vast.jrc.service.ec.europa.eu/sparql'
    VirtuosoUsername = 'dba'

    if KG_restriction:
        contextWordVirtuoso = ', '.join(sorted(KG_restriction))
    else:
        contextWordVirtuoso = ""

    # Check cache
    if cache_map_virtuoso is not None:
        if word in cache_map_virtuoso:
            if contextWordVirtuoso in cache_map_virtuoso[word]:
                entityBioeUrl = cache_map_virtuoso[word][contextWordVirtuoso]
                if DEBUG:
                    print("RETRIEVED CACHED RESULT FOR:\n", word, " => ", entityBioeUrl, "\n")
                # Retrieve cached exactURIs and relatedURIs
                if 'exactURIs' in cache_map_virtuoso[word]:
                    exactURIs = cache_map_virtuoso[word]['exactURIs']
                if 'relatedURIs' in cache_map_virtuoso[word]:
                    relatedURIs = cache_map_virtuoso[word]['relatedURIs']
                return entityBioeUrl, exactURIs, relatedURIs, cache_map_virtuoso

    # Build the exact SPARQL query
    if KG_restriction:
        from_clauses = ' '.join([f"FROM <{choice}>" for choice in KG_restriction])
    else:
        from_clauses = ""

    query_exact = f"""
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

    # Build fuzzy SPARQL queries using bif:contains (Virtuoso full-text index, much faster than regex)
    # Only used when exact query returns nothing.
    # Virtuoso requires a simple triple pattern before bif:contains (property paths don't work).
    # word_escaped: lowercase, strip all punctuation and parentheses, collapse whitespace
    word_escaped = re.sub(r'[^\w\s]', ' ', word.lower())
    word_escaped = re.sub(r'\s+', ' ', word_escaped).strip()

    # query_fuzzy_1: full phrase search (all tokens together)
    query_fuzzy_1 = f"""
        prefix skosxl: <http://www.w3.org/2008/05/skos-xl#>
        SELECT DISTINCT ?concept ?label
        {from_clauses}
        WHERE {{
          ?concept ?p ?label .
          FILTER (?p IN (skos:prefLabel, rdfs:label, skos:altLabel, skosxl:literalForm, obo:hasRelatedSynonym))
          ?label bif:contains "'{word_escaped}'" .
        }}
        LIMIT 10
    """

    # query_fuzzy_2: pairwise AND combinations of unigrams (length n-1)
    # e.g. "sudan virus sudv" → ("sudan" AND "virus") OR ("sudan" AND "sudv") OR ("virus" AND "sudv")
    tokens = word_escaped.split()
    query_fuzzy_2 = None
    if len(tokens) >= 2:
        combos = list(combinations(tokens, len(tokens) - 1))
        or_parts = []
        for combo in combos:
            and_clause = ' AND '.join([f'"{t}"' for t in combo])
            or_parts.append(f'({and_clause})')
        bif_contains_expr = ' OR '.join(or_parts)
        query_fuzzy_2 = f"""
        prefix skosxl: <http://www.w3.org/2008/05/skos-xl#>
        SELECT DISTINCT ?concept ?label
        {from_clauses}
        WHERE {{
          ?concept ?p ?label .
          FILTER (?p IN (skos:prefLabel, rdfs:label, skos:altLabel, skosxl:literalForm, obo:hasRelatedSynonym))
          ?label bif:contains '{bif_contains_expr}' .
        }}
        LIMIT 10
    """

    onto_clauses = ""
    if KG_restriction:
        for choice in KG_restriction:
            if choice == "SNOMED":
                choice = "SNOMEDCT"
            elif choice == "RO":
                choice = "OBOREL"
            elif choice == "TRANS":
                choice = "PTRANS"
            elif choice == "FoodOn":
                choice = "FOODON"
            elif choice == "GeoSPARQL":
                choice = "GEOSPARQL"
            elif choice == "NCBITaxon_":
                choice = "NCBITAXON"
            if choice in ONLY_Ontologies_OnBIOPORTAL:
                onto_clauses = onto_clauses + choice + ","
        if onto_clauses and onto_clauses[-1] == ",":
            onto_clauses = onto_clauses[:-1]
    else:
        onto_clauses = "AEO,BFO,BIM,BCGO,CL,CHIRO,CHEBI,DCM,FMA,GO,GENO,GEOSPARQL,HL7,DOID,HP,HP_O,IDO,IAO,ICD10,LOINC,MESH,MONDO,NCIT,NCBITAXON,NIFCELL,NIFSTD,GML,OBCS,OCHV,OHPI,OPB,PTRANS,PLOSTHES,RADLEX,OBOREL,STY,SO,SNOMEDCT,STATO,SYMP,FOODON,UBERON,VO"

    url = f"https://services.data.bioontology.org/annotatorplus/?text={word.lower()}&ontologies={onto_clauses}&longest_only=true&exclude_numbers=true&whole_word_only=true&exclude_synonyms=false&negation=false&experiencer=false&temporality=false&score_threshold=0&confidence_threshold=0&display_links=false&display_context=false&score=cvalue&apikey={key_bioportal}"

    try:

        if UseBioportalForLinking == False:
            # Use Virtuoso SPARQL endpoint (for KGs not available on BioPortal)
            if DEBUG:
                print("Use Virtuoso Sparql endpoint for linking (exact) ... " + word.lower())

            responseText = sparqlQuery(endpoint, query_exact, VirtuosoUsername, key_virtuoso, USE_CACHE)

            if responseText:
                results = json.loads(responseText)

                if len(results) > 0 and results['results']['bindings']:
                    entityBioeUrl = str(results['results']['bindings'][0]['concept']['value'])

                    if cache_map_virtuoso is not None:
                        if not word in cache_map_virtuoso:
                            cache_map_virtuoso[word] = {}
                        cache_map_virtuoso[word][contextWordVirtuoso] = entityBioeUrl

                    for result in results['results']['bindings']:
                        contextConcept = result['concept']['value']
                        if contextConcept not in exactURIs:
                            exactURIs.append(contextConcept)

                    exactURIs = list(set(exactURIs))

                    if cache_map_virtuoso is not None:
                        if not word in cache_map_virtuoso:
                            cache_map_virtuoso[word] = {}
                        cache_map_virtuoso[word]['exactURIs'] = exactURIs
                        cache_map_virtuoso[word]['relatedURIs'] = relatedURIs

                    return entityBioeUrl, exactURIs, relatedURIs, cache_map_virtuoso

            # No exact results from Virtuoso → try fuzzy queries (bif:contains)
            if not entityBioeUrl:
                if DEBUG:
                    print("Use Virtuoso Sparql endpoint for linking (fuzzy_1) ... " + word_escaped)

                fuzzy_found = False
                fuzzy_failed = False   # the endpoint errored: the outcome is unknown, not "no match"
                try:
                    responseText = sparqlQuery(endpoint, query_fuzzy_1, VirtuosoUsername, key_virtuoso, USE_CACHE)

                    if responseText:
                        results = json.loads(responseText)

                        if len(results) > 0 and results['results']['bindings']:
                            fuzzy_found = True
                            for result in results['results']['bindings']:
                                contextConcept = result['concept']['value']
                                if contextConcept not in relatedURIs:
                                    relatedURIs.append(contextConcept)

                            relatedURIs = list(set(relatedURIs))

                            if cache_map_virtuoso is not None:
                                if not word in cache_map_virtuoso:
                                    cache_map_virtuoso[word] = {}
                                cache_map_virtuoso[word][contextWordVirtuoso] = None
                                cache_map_virtuoso[word]['exactURIs'] = []
                                cache_map_virtuoso[word]['relatedURIs'] = relatedURIs

                            return None, [], relatedURIs, cache_map_virtuoso

                except Exception as err:
                    fuzzy_failed = True
                    if DEBUG:
                        print(f"Virtuoso fuzzy_1 SPARQL failed for '{word}': {err}")
                    logging.error(f'ERROR on Virtuoso fuzzy_1 linking for "{word.lower()}": {err}')

                # query_fuzzy_1 returned nothing → try query_fuzzy_2 (pairwise AND combinations)
                if not fuzzy_found and query_fuzzy_2 is not None:
                    if DEBUG:
                        print("Use Virtuoso Sparql endpoint for linking (fuzzy_2) ... " + word_escaped)

                    try:
                        responseText = sparqlQuery(endpoint, query_fuzzy_2, VirtuosoUsername, key_virtuoso, USE_CACHE)

                        if responseText:
                            results = json.loads(responseText)

                            if len(results) > 0 and results['results']['bindings']:
                                for result in results['results']['bindings']:
                                    contextConcept = result['concept']['value']
                                    if contextConcept not in relatedURIs:
                                        relatedURIs.append(contextConcept)

                                relatedURIs = list(set(relatedURIs))

                                if cache_map_virtuoso is not None:
                                    if not word in cache_map_virtuoso:
                                        cache_map_virtuoso[word] = {}
                                    cache_map_virtuoso[word][contextWordVirtuoso] = None
                                    cache_map_virtuoso[word]['exactURIs'] = []
                                    cache_map_virtuoso[word]['relatedURIs'] = relatedURIs

                                return None, [], relatedURIs, cache_map_virtuoso

                    except Exception as err:
                        fuzzy_failed = True
                        if DEBUG:
                            print(f"Virtuoso fuzzy_2 SPARQL failed for '{word}': {err}")
                        logging.error(f'ERROR on Virtuoso fuzzy_2 linking for "{word.lower()}": {err}')

                # Nothing found at all: cached as "no match", but only when the queries really did
                # answer. If one of them errored the term stays out of the cache, so that it is
                # looked up again on the next run instead of being unlinked for ever.
                if fuzzy_failed:
                    return entityBioeUrl, exactURIs, relatedURIs, cache_map_virtuoso

                if cache_map_virtuoso is not None:
                    if not word in cache_map_virtuoso:
                        cache_map_virtuoso[word] = {}
                    cache_map_virtuoso[word][contextWordVirtuoso] = None
                    cache_map_virtuoso[word]['exactURIs'] = []
                    cache_map_virtuoso[word]['relatedURIs'] = []

        else:
            # Use BioPortal API for linking (faster, all KGs are on BioPortal)
            if DEBUG:
                print("Use Bioportal for linking ... " + word.lower())

            is_annotatorplus_linking = "services.data.bioontology.org/annotatorplus" in url

            # an ApiRequestError here is left to propagate to the outer except: it means the API did
            # not answer, which is not a "no match" and must not be cached
            data = _api_get_json(url, DEBUG)

            if not data:
                if cache_map_virtuoso is not None:
                    if not word in cache_map_virtuoso:
                        cache_map_virtuoso[word] = {}
                    cache_map_virtuoso[word][contextWordVirtuoso] = None
                return None, [], [], cache_map_virtuoso

            dff = pd.DataFrame(data)
            dff = dff.drop(columns=['hierarchy', 'mappings'])

            expanded_annotated_class = pd.json_normalize(dff['annotatedClass'])
            expanded_annotations = pd.DataFrame(dff['annotations'].tolist(), index=dff.index)
            expanded_annotations = pd.json_normalize(expanded_annotations[0])

            df_expanded = dff.drop(columns=['annotatedClass', 'annotations']).join(expanded_annotated_class).join(
                expanded_annotations)

            if not is_annotatorplus_linking:
                if 'score' not in df_expanded.columns:
                    df_expanded['score'] = 10.0
                if 'prefLabel' not in df_expanded.columns:
                    if 'text' in df_expanded.columns:
                        df_expanded['prefLabel'] = df_expanded['text']
                    else:
                        df_expanded['prefLabel'] = None

            df_expanded['@id'] = df_expanded['@id'].str.replace(
                "http://purl.bioontology.org/ontology/SNOMEDCT/",
                "http://snomed.info/id/"
            )

            if not df_expanded.empty:
                df_expanded = df_expanded.sort_values(
                    by=['from', 'text', 'score', 'matchType'], ascending=[True, True, False, False])
                df_expanded = df_expanded.drop_duplicates(subset=['@id'])

                # Classify each result as exact or related
                for index, row in df_expanded.iterrows():
                    if row['@id'] is None or pd.isna(row['@id']):
                        continue
                    concept_uri = str(row['@id'])

                    is_exact = False
                    # Check prefLabel exact match
                    if 'prefLabel' in df_expanded.columns and isinstance(row.get('prefLabel'), str):
                        if row['prefLabel'].lower() == word.lower():
                            is_exact = True
                    # Check synonym exact match
                    if not is_exact and 'synonym' in df_expanded.columns:
                        syns = row.get('synonym')
                        if isinstance(syns, list) and any(s.lower() == word.lower() for s in syns):
                            is_exact = True

                    if is_exact:
                        if concept_uri not in exactURIs:
                            exactURIs.append(concept_uri)
                    else:
                        if concept_uri not in relatedURIs:
                            relatedURIs.append(concept_uri)

                # The top exact match becomes entityBioeUrl
                if exactURIs:
                    entityBioeUrl = exactURIs[0]

                exactURIs = list(set(exactURIs))
                relatedURIs = list(set(relatedURIs))

                if cache_map_virtuoso is not None:
                    if not word in cache_map_virtuoso:
                        cache_map_virtuoso[word] = {}
                    cache_map_virtuoso[word][contextWordVirtuoso] = entityBioeUrl
                    cache_map_virtuoso[word]['exactURIs'] = exactURIs
                    cache_map_virtuoso[word]['relatedURIs'] = relatedURIs

                return entityBioeUrl, exactURIs, relatedURIs, cache_map_virtuoso

            else:
                if cache_map_virtuoso is not None:
                    if not word in cache_map_virtuoso:
                        cache_map_virtuoso[word] = {}
                    cache_map_virtuoso[word][contextWordVirtuoso] = None
                return None, [], [], cache_map_virtuoso

    except Exception as err:
        # the query did not answer (timeout, throttling, endpoint down, malformed answer): nothing is
        # written to the cache, because this is not a "no match" and caching it would keep the term
        # unlinked in every run to come. A real absence of results is cached in the branches above.
        if DEBUG:
            print(f"Entity linking failed for '{word}': {err}")
        logging.error(f'ERROR on entity linking for "{word.lower()}": {err}')

    return entityBioeUrl, exactURIs, relatedURIs, cache_map_virtuoso


def process_row4Linking(row, rowindex, DEBUG, key_geonames, cache_map_geonames, cacheGeonames_filename, key_virtuoso, cache_map_virtuoso, cacheVirtuoso_filename, key_bioportal, USE_CACHE, KG_restriction=None, UseBioportalForLinking=True):
    """Entity linking using GeoNames for geo entities and Virtuoso+BioPortal for bio entities.

    Returns (result, exactURIs, relatedURIs, cache_map_geonames, cache_map_virtuoso):
      - result: the top entity URL (namedEntity)
      - exactURIs: URIs matched by exact prefLabel/altLabel/synonym → owl:sameAs
      - relatedURIs: URIs matched fuzzily → skos:related
    """
    result = ""
    exactURIs = []
    relatedURIs = []

    try:

        if row.empty:
            return result, exactURIs, relatedURIs, cache_map_geonames, cache_map_virtuoso

        if row['IsGeo'] == 1:

            if DEBUG:
                print(f"\n----- IsGeo ... COMPUTING {row['word']} ")

            result, ALLURIScontext, cache_map_geonames = geonames_api_call(row['word'], DEBUG, key_geonames, cache_map_geonames)
            # GeoNames results are all exact matches
            if ALLURIScontext and isinstance(ALLURIScontext, list):
                exactURIs = list(set(ALLURIScontext))

            if USE_CACHE:
                if (rowindex != 0) and (rowindex % 50 == 0):
                    if DEBUG:
                        print("saving geonames cache")
                    if cache_map_geonames is not None:
                        with open(cacheGeonames_filename, "w") as f:
                            json.dump(cache_map_geonames, f)

        elif row['IsBio'] == 1:

            if DEBUG:
                print(f"\n----- isBio COMPUTING ... {row['word']}")

            result, exactURIs, relatedURIs, cache_map_virtuoso = getUrlBioAndAllOtherBioConcepts(
                row['word'], DEBUG, key_virtuoso, cache_map_virtuoso, key_bioportal,
                KG_restriction=KG_restriction, USE_CACHE=USE_CACHE,
                UseBioportalForLinking=UseBioportalForLinking)

            if exactURIs and isinstance(exactURIs, list):
                exactURIs = list(set(exactURIs))
            if relatedURIs and isinstance(relatedURIs, list):
                relatedURIs = list(set(relatedURIs))

            if USE_CACHE:
                if (rowindex != 0) and (rowindex % 50 == 0):
                    if DEBUG:
                        print("saving virtuoso cache")
                    if cache_map_virtuoso is not None:
                        with open(cacheVirtuoso_filename, "w") as f:
                            json.dump(cache_map_virtuoso, f)

        return result, exactURIs, relatedURIs, cache_map_geonames, cache_map_virtuoso

    except Exception as e:
        return result, exactURIs, relatedURIs, cache_map_geonames, cache_map_virtuoso




if __name__ == "__main__":

    
    print("\nTransform Epidemic Extractions data to Epidemic Ontology and load into Knowledge-base Virtuoso\n\n") 
    
    startTime = time.time()

    ## Input parameters
    print("INPUT PARAMETERS\n\n")

    # %%
    DEBUG = True  #True  #False
    print("DEBUG = "+str(DEBUG)+"\n")

    # %%
    eKG_ontologyPREAMBLE_file_location = "./epidemicIE_PREAMBLE.ttl"
    print("eKG_ontology_file_location = "+str(eKG_ontologyPREAMBLE_file_location)+"\n")

    # %%
    input_data_epidemicextractions = "/OutputAnnotatedTexts-FT-LLM.csv"
    print("input_data_epidemicextractions = "+str(input_data_epidemicextractions)+"\n")

    # %%
    virtuoso_connection_parameters = 'https://api-vast.jrc.service.ec.europa.eu'
    print("virtuoso_connection_parameters = " + str(virtuoso_connection_parameters) + "\n")

    # %%
    virtuoso_key_file = "VIRTUOSO.key"
    if os.path.exists(virtuoso_key_file):
        with open(virtuoso_key_file) as f:
            virtuoso_password = f.read().strip()
    else:
        raise FileNotFoundError(f"Virtuoso key file not found: {virtuoso_key_file}")
    virtuoso_credentials = 'dba:' + virtuoso_password
    print("virtuoso_credentials = dba:***\n")

    # %%
    virtuoso_rdf_synk_location = '/DAV/home/dba/rdfimport/'
    print("virtuoso_rdf_synk_location = " + str(virtuoso_rdf_synk_location) + "\n")

    # %%
    PerformEntityLinking=True  #True False
    print("PerformEntityLinking = " + str(PerformEntityLinking) + "\n")

    # %%
    USE_CACHE=True
    print("USE_CACHE = " + str(USE_CACHE) + "\n")

    # %%
    geonameskey_filename = "GEONAMES-API.key" #file location where it is stored the geonames api key
    print("geonameskey_filename = " + str(geonameskey_filename) + "\n")

    # %%
    bioportalkey_filename = "NCBO-BioPortal.key"  #file location where it is stored the NCBO BioPortal api key
    print("bioportalkey_filename  = " + str(bioportalkey_filename ) + "\n")

    # %%
    # Add the --KG_restriction argument
    # Since nargs='+' is used, this will accept one or more values from the command line
    KGchoices = POSSIBLE_KGchoices_List  # this to link using virtuoso
    #KGchoices = ONLY_Ontologies_OnBIOPORTAL # this to link using the bioportal only
    # KGchoices = ['SNOMED', 'LOINC', 'ICD10', 'NCIT']
    # KGchoices = ['SNOMED', 'LOINC', 'ICD10', 'MESH', 'NCIT']  # restricts the input to these values only
    # KGchoices = ['SNOMED', 'LOINC', 'ICD10', 'MESH', 'NCIT', 'MONDO', 'DOID', 'HP', 'CHEBI', 'NCBITAXON']
    if KGchoices:
        KGchoices.sort()
    KG_restriction = KGchoices
    print("KG_restriction = " + str(KG_restriction) + "\n")


    #################################################################################################################


    # Start script
    print("\nSTART SCRIPT\n\n")

    eKG_ontology_file_location = writeOntoFile(DEBUG,eKG_ontologyPREAMBLE_file_location,input_data_epidemicextractions,PerformEntityLinking, USE_CACHE, geonameskey_filename, bioportalkey_filename, KG_restriction=KG_restriction)

    if eKG_ontology_file_location:

        # Convert the Turtle file to RDF
        output_rdf_file = eKG_ontology_file_location.replace(".ttl", ".rdf")
        convert_turtle_to_rdf(DEBUG, eKG_ontology_file_location, output_rdf_file)

        ### UPLOAD AND WRITE ONTOLOGY FILE TO VIRTUOSO KNOWLEDGE-BASE
        isUploaded = WebDavUpload(DEBUG,eKG_ontology_file_location,virtuoso_connection_parameters,virtuoso_rdf_synk_location,virtuoso_credentials)
        if isUploaded:
            if DEBUG:
                print("File "+eKG_ontology_file_location+" correctly uploaded into Virtuoso WebDav: "+str(virtuoso_connection_parameters)+str(virtuoso_rdf_synk_location))


    endend = time.time()
    hours, rem = divmod(endend - startTime, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total Computational Time: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes),seconds) + "\n")

    print("End script")
    



