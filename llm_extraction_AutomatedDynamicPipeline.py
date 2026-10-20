"""Core daily pipeline of the epidemiological information-extraction system.

This module is the heart of the daily-updated epidemiological knowledge graph (eKG). For every
not-yet-computed day it:

  1. resolves the resume point (the first day that has no ontology on disk yet);
  2. acquires the WHO/EIOS items of each day through the EIOS API (``eios_client``);
  3. pre-processes and de-duplicates the raw articles;
  4. runs the fine-tuned open-weight extraction model (EpiQwen3.5-9B) locally to turn each
     outbreak report into structured fields;
  5. builds the per-day ontology on the European Crisis Management Ontology (ECMO) with entity
     linking of diseases and places (``ONTO-RESOURCES_eKG``);
  6. rebuilds the overall extraction files and the overall ontology, including the shareable
     "_red" versions that are published to the JRC Research Data repository.

Entry point: ``main(working_directory)``. It is normally driven by the scheduler in
``SCHEDULE-RUN_AutomatedDynamicPipeline.py``.

Credentials (EIOS, GeoNames, BioPortal, Virtuoso, GPT API tokens) are never hardcoded: they are
read at run time from a local ``config.py`` and from ``*.key`` files in the script directory. See
``config.example.py`` and ``keys/README.md``.
"""

import os
# The Hugging Face cache location can be set here if the default (~/.cache/huggingface) is not
# suitable, e.g. on a shared machine with limited home quota. Left to the HF default by design;
# set the HF_HOME / HF_HUB_CACHE environment variables before running if you need to relocate it.
#   os.environ["HF_HOME"] = "/path/to/huggingface/cache"

from pathlib import Path
import sys
import glob
import pandas as pd
import numpy as np
import datetime
import logging
import time
import json
import gc
from dateutil.parser import isoparse
from rdflib import Graph, RDF, URIRef

from transformers import AutoTokenizer
import torch
from tqdm import tqdm

# NOTE: the former imports of the 3-LLM ensemble stack (llmquery, epidemicExtractor_Deployment,
# llm_extraction_Dictionary_Deployment, llm_extraction_majorityVoting_Deployment) have been
# dropped: the extraction is now done by the fine-tuned model, see further down.

import importlib
_onto_module = importlib.import_module("ONTO-RESOURCES_eKG")
writeOntoFile = _onto_module.writeOntoFile
convert_turtle_to_rdf = _onto_module.convert_turtle_to_rdf
record_ids_of_extractions = _onto_module.record_ids_of_extractions
POSSIBLE_KGchoices_List = _onto_module.POSSIBLE_KGchoices_List
ONLY_Ontologies_OnBIOPORTAL = _onto_module.ONLY_Ontologies_OnBIOPORTAL

# KG_restriction for entity linking in ontology generation
# Since nargs='+' is used in standalone, this accepts one or more values
KGchoices = POSSIBLE_KGchoices_List  # this to link using virtuoso
#KGchoices = ONLY_Ontologies_OnBIOPORTAL # this to link using the bioportal only
# KGchoices = ['SNOMED', 'LOINC', 'ICD10', 'NCIT']
# KGchoices = ['SNOMED', 'LOINC', 'ICD10', 'MESH', 'NCIT']  # restricts the input to these values only
# KGchoices = ['SNOMED', 'LOINC', 'ICD10', 'MESH', 'NCIT', 'MONDO', 'DOID', 'HP', 'CHEBI', 'NCBITAXON']
if KGchoices:
    KGchoices.sort()
KG_restriction = KGchoices
print("KG_restriction = " + str(KG_restriction) + "\n")


############################################################################################
### DATA ACQUISITION SETTINGS
############################################################################################

# First day of the whole series, in "YYYYMMDD" format: the day the day-by-day EIOS harvesting
# starts from when nothing has been computed yet. It is only the fallback of the resume point,
# which is determined automatically at every run by resolve_starting_date(), see below.
STARTING_DATE = "20230501"

# Name of the ontology written for every harvested day in its own "YYYYMMDD" folder. It is the
# last artefact produced for a day, so its presence is what marks that day as fully computed:
# resolve_starting_date() looks for it to determine from which day the harvesting has to resume.
ITER_ONTOLOGY_BASENAME = "epidemicIE_IterUpdate.ttl"

# Name of the extraction file written for every harvested day in its own "YYYYMMDD" folder. The two
# overall extraction files are rebuilt from all of them at every run, see
# rebuild_overall_extractions().
ITER_EXTRACTIONS_BASENAME = "OutputAnnotatedTexts-FT-LLM-IterUpdate.csv"

# The per-day disk cache of the raw getMedicalOfficialArticles results lives in the working
# directory of the run, i.e. the `synk_location` given to main() (see __main__ below), so it is
# NOT configured here. One subfolder per harvested day, named "YYYYMMDD", containing:
#   articles.json  -> the raw response of getMedicalOfficialArticles for that day
#   fulltexts.json -> {item_id: text} as returned by get_item_details for that day
#   _COMPLETE.json -> written last, only when the day has been fully retrieved.
#                     Its presence is what marks the day as cached: a day whose folder
#                     exists but has no _COMPLETE.json is considered partial and is resumed
#                     (only the still-missing item texts are re-requested).
# The same "YYYYMMDD" folder later receives the extractions of the day and its ontology, see main().
MEDICAL_OFFICIAL_CACHE_COMPLETE_MARKER = "_COMPLETE.json"

# Prefix used to build the 'fileid' of the harvested articles (it replaces the former
# "TOTAL_ProMED/" and "who_dons/" prefixes).
MEDICAL_OFFICIAL_FILEID_PREFIX = "MedicalOfficial/"

############################################################################################
### SERVICE AVAILABILITY WINDOW
###
### The web services the pipeline depends on (in particular the Virtuoso SPARQL endpoint used by
### the entity linking of the ontology generation) are only up on working days, roughly 08:00-19:30.
### Outside that window every linking query fails, so no work must be started: the pipeline stops
### where it is and the next scheduled run resumes from there (a day is only marked as computed once
### its own ontology is on disk, see ITER_ONTOLOGY_BASENAME / resolve_starting_date()).
### The bounds are local wall-clock times, i.e. the same clock as the "Europe/Amsterdam" schedule of
### SCHEDULE-RUN_AutomatedDynamicPipeline.py (this machine runs on Europe/Rome, same CET/CEST).
############################################################################################

SERVICE_WINDOW_START = datetime.time(8, 15)
SERVICE_WINDOW_END = datetime.time(19, 0)

# Safety margin, in minutes, required before starting the ontology of one more day: the entity
# linking of a single day takes around half an hour (a bit more when few terms are in cache), so a day
# started right before the closing time would run into the dead endpoint half way through and produce
# an incomplete ontology, which - being written on disk - would then count as a computed day.
# A day is therefore only started when at least this many minutes of window are left; the days left
# out keep their extractions and are computed at the next run.
SERVICE_WINDOW_ONTOLOGY_MARGIN_MINUTES = 45


def within_service_window(now=None, margin_minutes=0):
    """True when the services are expected to be up: a working day, between the window bounds.

    `margin_minutes` requires the window to stay open for at least that many minutes more, for the
    callers that are about to start a long piece of work (see SERVICE_WINDOW_ONTOLOGY_MARGIN_MINUTES).
    """
    if now is None:
        now = datetime.datetime.now()

    if now.weekday() >= 5 or now.time() < SERVICE_WINDOW_START:
        return False

    closing_time = datetime.datetime.combine(now.date(), SERVICE_WINDOW_END)

    return now <= closing_time - datetime.timedelta(minutes=margin_minutes)


def service_window_message(now=None, margin_minutes=0):
    """One-line description of the current position with respect to the service window."""
    if now is None:
        now = datetime.datetime.now()
    return (now.strftime("%A %H:%M") + " is " +
            ("inside" if within_service_window(now, margin_minutes) else "outside") +
            " the service window (working days " + SERVICE_WINDOW_START.strftime("%H:%M") + "-" +
            SERVICE_WINDOW_END.strftime("%H:%M") +
            (", with the " + str(margin_minutes) + " min safety margin before the closing time"
             if margin_minutes else "") + ")")


############################################################################################
### FINE-TUNED EXTRACTION MODEL SETTINGS
###
### The extraction is done by the fine-tuned model jattokatarratto/EpiQwen3.5-9B (a PEFT/LoRA
### adapter on top of Qwen/Qwen3.5-9B), exactly as in causal_LLM_inference-Run-EPIextraction.py.
### It replaces the former 3-LLM ensemble + vocabulary/majority-voting pipeline, and, since
### 2026-10-12, its own predecessor AI4PH/EpiQwen2.5-7B.
###
### EpiQwen3.5-9B is served differently from EpiQwen2.5-7B: it was trained on the chat template of
### its base model with thinking disabled, so the prompt is rendered by that template, generation
### stops on the chat stop token <|im_end|> and the weights are kept in bf16 (the adapter merged
### into the base model) instead of being quantized to 4 bit. None of this is hardcoded here: the
### model repository ships a serving_prompt_format.json recording the format it was trained with,
### and serving_format.py (shared with the donWHO_iCL fine-tuning project) applies it. A model
### without that file, like EpiQwen2.5-7B, is served as before, with the Alpaca-style prompt of
### causal_LLM_inference-Run-EPIextraction.build_prompt() and 4-bit weights: switching back to it
### only means putting its identifiers below.
###
### It needs transformers >= 5.17.0 (for Qwen3_5ForConditionalGeneration), peft >= 0.21.0 and the
### matching torch, see the model card in README.md.
############################################################################################

FT_LOAD_FROM = "huggingface"  # "huggingface" (adapter from the HF hub) or "disk" (local adapter)
FT_MODEL_HF_ID = "jattokatarratto/EpiQwen3.5-9B"
FT_BASE_MODEL = "Qwen/Qwen3.5-9B"
# Local copy of the adapter, used only when FT_LOAD_FROM is "disk": the directory of the fine-tuned
# model (a merged model or an adapter directory, e.g. the snapshot of the HF cache). Left empty
# because the pipeline loads from the hub; the hub copy is cached by huggingface_hub anyway.
FT_MODEL_PATH = ""
# Answer budget. The generation stops on the chat stop token of the model, so this is only a cap:
# it is the value EpiQwen3.5-9B was trained and evaluated with (its serving_prompt_format.json
# recommends the same, and is given precedence when it carries one).
FT_MAX_NEW_TOKENS = 2048
FT_SEED = 41
# Safety cap on the prompt length: the inference script uses truncation=False, so a very long
# full-text article would otherwise blow up the generation. The text of articles longer than this
# is truncated (with a printed warning) instead of being skipped.
FT_MAX_INPUT_TOKENS = 8192
# Weights in 4 bit (NF4) instead of bf16. "auto" follows the model: 4 bit for the legacy
# Alpaca-format fine-tunes, as they were always run, bf16 for the chat-format ones, which are
# merged with their adapter and were evaluated that way. True / False force it.
FT_LOAD_4BIT = "auto"

# When True, if the extractions of the current iteration are already on disk (the
# OutputAnnotatedTexts-FT-LLM-IterUpdate.csv of the iteration folder) and they cover exactly the
# articles that have just been harvested, they are read back instead of being recomputed: a re-run
# after a failure of (or a fix to) the ontology generation does not pay the GPU time again.
# Set it to False to always run the model.
FT_REUSE_ITER_EXTRACTIONS = True

# The granular fields extracted from jattokatarratto/EpiQwen3.5-9B, in the model's own order.
# This is an ALLOW-LIST: each field below becomes a column of
# OutputAnnotatedTexts-FT-LLM-IterUpdate.csv, and anything else the model may emit (e.g.
# "DiseaseLevel3") is simply not kept. Nothing is really lost, since the verbatim model output
# is always stored in the 'json_original' column.
FT_EXTRACTION_FIELDS = [
    "DiseaseLevel1",
    "DiseaseLevel2",
    "Country",
    "ISO",
    "OutbreakEpicenter",
    "CasesTotal",
    "CasesSuspected",
    "CasesProbable",
    "CasesConfirmed",
    "Deaths",
    "OutbreakStartYear",
    "OutbreakStartMonth",
    "OutbreakStartDay",
    "OutbreakDetectionYear",
    "OutbreakDetectionMonth",
    "OutbreakDetectionDay",
    "OutbreakVerificationYear",
    "OutbreakVerificationMonth",
    "OutbreakVerificationDay",
    "OutbreakEnd",
    "OutbreakEndYear",
    "OutbreakEndMonth",
    "OutbreakEndDay",
]

# The metadata of the media outlet the article comes from, as the EIOS api returns it in the
# "source" block of every item ("name", "id" and "url"). They travel with the article through the
# extractions and end up on the ekg:MediaReport node of the ontology.
FT_SOURCE_FIELDS = ['source_name_EIOS', 'source_id_EIOS', 'source_url_EIOS']

# The columns describing the article itself, prepended to the extraction fields.
FT_ARTICLE_FIELDS = (['fileid'] + FT_SOURCE_FIELDS +
                     ['date_pubDate_EIOS', 'date_processedOnDate_EIOS', 'date_cases_IMPUTED',
                      'texts', 'texts_length', 'outbreak_index', 'n_outbreaks'])

# The column joining a row of the overall extractions to its outbreak record in the ontology: it
# carries the identifier the record is published with, i.e. the rdfs:label of its ekg:OutbreakEvent
# ("recordYYYYMMDD_<i>"). It is not stored in the per-day extraction files: it is computed when the
# overall files are rebuilt (see rebuild_overall_extractions), out of the very rule the ontology
# writer uses to number the records.
RECORD_ID_FIELD = 'record_id'

# What the shareable version of the overall extractions does not carry: the full text of the article
# (copyright of the MedicalOfficial sources), everything identifying where the article comes from,
# i.e. its fileid and the metadata of its media outlet, and the raw answer of the model, which quotes
# the article. RECORD_ID_FIELD replaces them as the key of the row, exactly as the shareable ontology
# keeps the rdfs:label of the records and drops their ekg:MediaReport (see write_reduced_ontology).
# The rows that produced no outbreak record are left out of it as well.
REDUCED_EXTRACTIONS_DROPPED_FIELDS = ['texts', 'fileid', 'json_original'] + FT_SOURCE_FIELDS

# What the shareable version of the ontology does not carry, written as it appears in the Turtle (the
# reduction is done on the text of the complete file, see write_reduced_ontology): the nodes
# describing the article an outbreak comes from, the statements of the events pointing at them, and
# the identifier of the events, which is the identifier of that very article.
REDUCED_ONTOLOGY_DROPPED_CLASS = 'ekg:MediaReport'
REDUCED_ONTOLOGY_DROPPED_PREDICATES = ['foundation:hasSignal']
REDUCED_ONTOLOGY_EVENT_CLASS = 'ekg:OutbreakEvent'
REDUCED_ONTOLOGY_DROPPED_EVENT_PREDICATES = ['dcterms:identifier']

############################################################################################

import warnings
warnings.filterwarnings('ignore', message='TypedStorage is deprecated.')


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
        timeout=1200,  #5 min
    )


def get_EIOS_test_client(api_version: str = "1.0"):
    """Create and return an EIOSClient instance pointing to the test environment."""
    import config
    from eios_client import EIOSClient
    return EIOSClient(
        tenant_id=config.eios_tenant_id,
        client_id=config.eios_client_id,
        client_secret=config.eios_client_secret,
        scope=config.eios_scope,
        base_url=EIOSClient.DEFAULT_TEST_URL,
        api_version=api_version,
        timeout=1200,  #5 min
    )

def eios_call_with_retry(func, max_retries=50, sleep_seconds=2):
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


# Adapt date format to include seconds, milliseconds, and Z timezone
def adapt_date_format(date_str):
    if date_str.endswith('Z'):
        return date_str
    if '.' not in date_str:
        return date_str + ":00.000Z"
    return date_str + "Z"
    
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


def getMedicalOfficialArticles(startdate_ts_string, enddate_ts_string, eios_client, start=0, limit=100):
    """Get English Medical Official articles from EIOS.

    Confirmed working filter properties:
      source.subject     -> "Medical Official"
      source.type        -> "Medical"
      languageIso        -> "en" (article language)
      source.languageCode -> "en" (source language)

    `start`/`limit` are the pagination offset and page size: the defaults (0, 100) give exactly
    the single-page behaviour that was validated in eios_api_test_extractions.py. Note that the
    response 'count' is the TOTAL number of matching articles, which can be larger than the page
    size -> see getMedicalOfficialArticlesAllPages() to retrieve them all.
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
        "start": start,
        "limit": limit,
        "sorts": [
            {
                "property": "processedOnDate",
                "direction": "desc"
            }
        ]
    }
    result = eios_call_with_retry(lambda: eios_client.filter_items(query))
    return result


def getMedicalOfficialArticlesAllPages(startdate_ts_string, enddate_ts_string, eios_client,
                                       page_size=100, max_pages=100, DEBUG=False):
    """getMedicalOfficialArticles over all the pages of the result set.

    A single filter_items call returns at most `page_size` articles, while the returned 'count'
    is the total number of matching ones (e.g. 140 matches but only 100 returned for a single
    day), so without paging part of the day would be silently dropped. Pages are requested by
    increasing the 'start' offset, and merged (de-duplicated on the item id) into one response
    with the same {'count': ..., 'result': [...]} shape as the single-page call.

    Returns None if the very first request failed, so that the day is not cached and gets retried.
    """
    merged_result = []
    seen_ids = set()
    total_count = None

    for page in range(max_pages):
        page_json = getMedicalOfficialArticles(startdate_ts_string, enddate_ts_string, eios_client,
                                               start=page * page_size, limit=page_size)
        if not page_json:
            if page == 0:
                return None
            print("   Warning: page " + str(page) + " failed for the window " + str(startdate_ts_string) +
                  " - " + str(enddate_ts_string) + " -> keeping the " + str(len(merged_result)) + " articles retrieved so far")
            break

        if total_count is None:
            total_count = page_json.get("count", 0) or 0

        page_items = page_json.get("result") or []
        for item in page_items:
            item_id = str(item.get('id', '')) or None
            if item_id and item_id in seen_ids:
                continue
            if item_id:
                seen_ids.add(item_id)
            merged_result.append(item)

        if DEBUG and total_count > page_size:
            print("   [paging] page " + str(page) + ": +" + str(len(page_items)) + " items -> " +
                  str(len(merged_result)) + "/" + str(total_count))

        if len(page_items) < page_size or len(merged_result) >= total_count:
            break

    return {"count": total_count or 0, "result": merged_result}


############################################################################################
### PER-DAY DISK CACHE OF THE RAW getMedicalOfficialArticles RESULTS
############################################################################################

def extract_item_text(item_detail):
    """Build the text of an EIOS item, with the usual fallback chain.

    fullText -> translatedText -> title + pubDate + description + abstractiveSummary
    """
    if not item_detail:
        return ""
    full_text = item_detail.get('fullText')
    if full_text:
        return full_text
    return (item_detail.get('translatedText')
            or ((item_detail.get('title') or '') + ', ' + (item_detail.get('pubDate') or '') + ', ' +
                (item_detail.get('description') or '') + ', ' +
                (item_detail.get('abstractiveSummary') or '')).strip()
            or '')


def read_json_file(file_path, default=None):
    """Read a json file, returning `default` if it does not exist or cannot be parsed."""
    try:
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as fp:
                return json.load(fp)
    except Exception as err:
        print(f"   Warning: could not read cache file {file_path}: {err}")
    return default


def write_json_file(file_path, payload):
    """Write a json file atomically (write to .tmp then rename)."""
    tmp_path = file_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False)
    os.replace(tmp_path, file_path)


def getMedicalOfficialArticlesCached(day_key, startdate_ts_string, enddate_ts_string, eios_client,
                                     cache_root, DEBUG=False):
    """getMedicalOfficialArticles + get_item_details for one day, backed by a disk cache.

    `cache_root` is the root of the cache (the working directory of the run) and `day_key` is the
    day in "YYYYMMDD" format, which gives the name of the cache subfolder.

    If <cache_root>/<day_key>/_COMPLETE.json exists, nothing is requested to EIOS: the stored
    articles.json and fulltexts.json are read back from disk. Otherwise the day is retrieved
    (re-using any text already stored by a previous interrupted run) and then written to disk,
    with the _COMPLETE.json marker written last.

    Returns (articlesJson, fulltexts_by_item_id, from_cache).
    """
    day_dir = os.path.join(cache_root, day_key)
    articles_path = os.path.join(day_dir, "articles.json")
    fulltexts_path = os.path.join(day_dir, "fulltexts.json")
    complete_path = os.path.join(day_dir, MEDICAL_OFFICIAL_CACHE_COMPLETE_MARKER)

    # ---- 1) fully cached day: just read it back
    if os.path.isfile(complete_path):
        articlesJson = read_json_file(articles_path, default=None)
        fulltexts = read_json_file(fulltexts_path, default={})
        if articlesJson is not None:
            if DEBUG:
                print("   [cache] day " + day_key + " read from disk (" +
                      str(articlesJson.get("count", 0)) + " items, " + str(len(fulltexts)) + " texts)")
            return articlesJson, fulltexts, True
        print("   Warning: cache marker present but articles.json unreadable for day " + day_key +
              " -> re-fetching from EIOS")

    # ---- 2) not (or only partially) cached: call the API (all pages of the day)
    articlesJson = getMedicalOfficialArticlesAllPages(startdate_ts_string, enddate_ts_string,
                                                      eios_client, DEBUG=DEBUG)
    if not articlesJson:
        # request failed: do not cache anything, the day will be retried on the next run
        print("   Warning: getMedicalOfficialArticles returned nothing for day " + day_key +
              " -> day not cached, it will be retried")
        return articlesJson, {}, False

    # re-use the texts already downloaded by a previous interrupted run of this same day
    fulltexts = read_json_file(fulltexts_path, default={}) or {}

    all_texts_retrieved = True
    for YY in (articlesJson.get("result") or []):
        item_id = YY.get('id', '')
        if not item_id:
            continue
        item_id = str(item_id)
        if fulltexts.get(item_id):
            continue  # already downloaded
        try:
            item_detail = eios_client.get_item_details(item_id)
            text = extract_item_text(item_detail)
            if text:
                fulltexts[item_id] = text
            else:
                all_texts_retrieved = False
        except Exception as err:
            print(f"   Warning: could not fetch fullText for {item_id}: {err}")
            all_texts_retrieved = False

    # ---- 3) store the day on disk
    try:
        os.makedirs(day_dir, exist_ok=True)
        write_json_file(articles_path, articlesJson)
        write_json_file(fulltexts_path, fulltexts)
        if all_texts_retrieved:
            write_json_file(complete_path, {
                "day": day_key,
                "window": [startdate_ts_string, enddate_ts_string],
                "count": articlesJson.get("count", 0),
                "n_items": len(articlesJson.get("result") or []),
                "n_texts": len(fulltexts),
                "completed_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
        else:
            print("   Note: day " + day_key + " stored but incomplete (some item texts are missing)"
                  " -> it will be resumed on the next run")
    except Exception as err:
        print(f"   Warning: could not write the cache for day {day_key}: {err}")

    return articlesJson, fulltexts, False


############################################################################################
### EXTRACTION WITH THE FINE-TUNED MODEL jattokatarratto/EpiQwen3.5-9B
############################################################################################

import re
import ast

# the prompt building / cleaning / generation helpers are reused as they are from the
# inference script (loaded via importlib because of the hyphens in the file name)
_ft_module = importlib.import_module("causal_LLM_inference-Run-EPIextraction")
ft_build_prompt = _ft_module.build_prompt  # it already applies clean_input_text() internally
ft_generate_response = _ft_module.generate_response
ft_load_causal_lm = _ft_module.load_causal_lm
# the prompt format the model was trained with, see the comment on the FT_* constants above
import serving_format


def load_finetuned_epi_model(load_from=FT_LOAD_FROM, hf_model_id=FT_MODEL_HF_ID, base_model=FT_BASE_MODEL,
                             model_path=FT_MODEL_PATH, seed=FT_SEED, load_4bit=FT_LOAD_4BIT):
    """Load jattokatarratto/EpiQwen3.5-9B (base model + PEFT adapter), as in the inference script.

    Returns (model, tokenizer, fmt), where fmt is the serving prompt format of the model
    (None for a legacy Alpaca-prompt fine-tune such as AI4PH/EpiQwen2.5-7B); it has to be
    handed over to the prompt building and the generation, so that the model is served with
    the very prompt and stop token it was trained with.
    """
    import random
    from transformers import BitsAndBytesConfig
    from peft import PeftModel

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    adapter_location = hf_model_id if load_from == "huggingface" else model_path
    fmt = serving_format.find_serving_format(adapter_location)
    # a merged / full model has no adapter_config.json: then it is loaded on its own
    adapter_base = serving_format.adapter_base_model(adapter_location)
    if adapter_base is not None and not base_model:
        base_model = fmt["model_id"] if fmt is not None else adapter_base
    if load_4bit == "auto":
        # the chat-format fine-tunes were evaluated in bf16 with the adapter merged in
        load_4bit = fmt is None
    print("Prompt format: " + ("chat template (" + fmt["model_id"] + ")" if fmt is not None
                               else "Alpaca ### Instruction (legacy)")
          + " - weights in " + ("4-bit" if load_4bit else "bf16"))

    if fmt is not None:
        # The base model must be loaded with the very class it was trained with, which means a
        # transformers recent enough to know it (Qwen3_5ForConditionalGeneration for EpiQwen3.5-9B,
        # transformers >= 5.17.0). Without it transformers would either refuse the config or fall
        # back to another class, whose parameter names do not match the adapter: the extraction
        # would then run on a model the adapter has not really been applied to, which is worse
        # than not running at all, so this is checked up front.
        import transformers
        from transformers import AutoConfig
        try:
            architecture = (AutoConfig.from_pretrained(base_model, trust_remote_code=True)
                            .architectures or [None])[0]
        except Exception as err:
            architecture, config_error = None, err
        else:
            config_error = None
        if architecture is None or not hasattr(transformers, architecture):
            raise RuntimeError(
                "the fine-tuned extraction model " + str(adapter_location) + " needs its base model "
                + str(base_model) + " to be loaded with the class it was trained with, which this "
                "transformers " + transformers.__version__ + " does not provide"
                + (" (" + str(config_error) + ")" if config_error is not None
                   else " (" + str(architecture) + ")")
                + ". Upgrade the environment to the framework versions of the model card (README.md: "
                "transformers 5.17.0, torch 2.14.0, peft 0.21.0, accelerate 1.15.0).")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # the chat-format fine-tunes ship their own tokenizer (with the chat template they were
    # trained on); the legacy adapters use the tokenizer of their base model, as before
    tokenizer_src = base_model if (adapter_base is not None and fmt is None) else adapter_location
    print("Loading tokenizer from: " + str(tokenizer_src))
    ft_tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if fmt is None:
        ft_tokenizer.pad_token = ft_tokenizer.eos_token  # chat-format fine-tunes keep their own pad
    ft_tokenizer.padding_side = "left"

    weights_src = base_model if adapter_base is not None else adapter_location
    print("Loading " + ("base " if adapter_base is not None else "") + "model: " + str(weights_src))
    ft_model = ft_load_causal_lm(
        weights_src, fmt,
        quantization_config=bnb_config if load_4bit else None,
        device_map="auto",
        use_safetensors=True,
        trust_remote_code=True,
    )

    if adapter_base is not None:
        print("Loading PEFT adapter from " + str(load_from) + ": " + str(adapter_location))
        ft_model = PeftModel.from_pretrained(ft_model, adapter_location, torch_dtype=torch.bfloat16)
        if fmt is not None and not load_4bit:
            ft_model = ft_model.merge_and_unload()  # same weights, faster generation
    ft_model.eval()

    print("Fine-tuned extraction model loaded successfully!")
    return ft_model, ft_tokenizer, fmt


def ft_nan_to_null(json_like_string):
    """The model emits bare NaN tokens, which are not valid JSON -> turn them into null."""
    return re.sub(r'\bNaN\b', 'null', json_like_string)


def ft_normalize_value(value):
    """Normalise a single extracted value: every flavour of 'missing' becomes NaN."""
    if value is None:
        return np.nan
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in ("nan", "none", "null", "n/a", "na", "-", "unknown"):
            return np.nan
        return stripped
    return value


def ft_collect_objects(parsed):
    """Walk a parsed structure (dict / list / json-string) and collect the extraction dicts."""
    collected = []
    if isinstance(parsed, dict):
        collected.append(parsed)
    elif isinstance(parsed, (list, tuple)):
        for element in parsed:
            collected.extend(ft_collect_objects(element))
    elif isinstance(parsed, str):
        try:
            collected.extend(ft_collect_objects(json.loads(ft_nan_to_null(parsed))))
        except Exception:
            pass
    return collected


def parse_ft_extraction(raw_response):
    """Parse the raw output of jattokatarratto/EpiQwen3.5-9B into a list of extraction dicts.

    The model returns a list with one JSON object per outbreak (an empty list when the article
    reports none), using bare NaN for the missing fields, e.g.:

      ['{\n  "DiseaseLevel1": "Meningococcal disease",\n  "DiseaseLevel2": NaN, ... }']

    which is a python-literal list of JSON strings rather than plain JSON, so three decoding
    strategies are attempted in turn: python literal, plain JSON, and finally a regex sweep of
    the flat {...} blocks (which also survives any trailing text after the list).

    Returns a list of dicts restricted to FT_EXTRACTION_FIELDS, with missing values as NaN.
    """
    if raw_response is None:
        return []

    text = str(raw_response).strip()
    if not text:
        return []

    # strip an eventual markdown code fence
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    objects = []

    # A) python literal: handles the list-of-json-strings shape the model actually emits
    try:
        objects = ft_collect_objects(ast.literal_eval(text))
    except Exception:
        objects = []

    # B) plain JSON, once the bare NaN tokens are turned into null
    if not objects:
        try:
            objects = ft_collect_objects(json.loads(ft_nan_to_null(text)))
        except Exception:
            objects = []

    # C) last resort: sweep the flat {...} blocks one by one
    if not objects:
        for block in re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL):
            try:
                candidate = json.loads(ft_nan_to_null(block))
                if isinstance(candidate, dict):
                    objects.append(candidate)
            except Exception:
                continue

    # FT_EXTRACTION_FIELDS is an allow-list: only those fields are kept (normalising the missing
    # ones to NaN), anything else the model emits is discarded. Objects that end up completely
    # empty are dropped.
    extractions = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        row = {field: ft_normalize_value(obj.get(field)) for field in FT_EXTRACTION_FIELDS}
        if any(not (isinstance(v, float) and np.isnan(v)) for v in row.values()):
            extractions.append(row)

    return extractions


def ft_make_prompt(ft_tokenizer, ft_fmt, article_text, max_input_tokens=FT_MAX_INPUT_TOKENS,
                   fileid=""):
    """Render the extraction prompt of one article, truncating the article text if need be.

    Returns (prompt, was_truncated). The prompt is the one the model was fine-tuned on: the chat
    template of its base model when the model ships a serving format (jattokatarratto/EpiQwen3.5-9B),
    the Alpaca "### Instruction" prompt otherwise (AI4PH/EpiQwen2.5-7B).

    The cap is applied to the ARTICLE TEXT, not to the rendered prompt: the rendered prompt ends
    with the instruction and the generation prompt of the chat template, which must stay intact
    (cutting them would leave the model with a prompt it has never seen, and no text to answer
    about). The text is therefore shortened and the prompt rendered again, until it fits.
    """
    def render(text):
        if ft_fmt is None:
            return ft_build_prompt(text)
        return serving_format.build_chat_prompt(ft_tokenizer, ft_fmt, text)

    def n_tokens(prompt):
        return len(ft_tokenizer(prompt, return_tensors=None, truncation=False, padding=False,
                                add_special_tokens=ft_fmt is None)['input_ids'])

    prompt = render(article_text)
    if not max_input_tokens or n_tokens(prompt) <= max_input_tokens:
        return prompt, False

    # the overhead of the prompt around the text is constant: measure it once, then cut the text
    # to the remaining budget (in tokens of the article text) and re-render
    n_full = n_tokens(prompt)
    overhead = n_tokens(render(""))
    text_ids = ft_tokenizer(serving_format.clean_input_text(str(article_text)), return_tensors=None,
                            truncation=False, padding=False, add_special_tokens=False)['input_ids']
    budget = max(max_input_tokens - overhead, 0)
    while True:
        text_cut = ft_tokenizer.decode(text_ids[:budget], skip_special_tokens=True)
        prompt = render(text_cut)
        if budget <= 0 or n_tokens(prompt) <= max_input_tokens:
            break
        budget -= max(1, (n_tokens(prompt) - max_input_tokens))  # re-tokenisation may add a token

    print("   Warning: prompt of " + str(n_full) + " tokens for " + str(fileid) +
          " -> article text truncated, prompt down to " + str(n_tokens(prompt)) + " tokens" +
          " (cap " + str(max_input_tokens) + ")")
    return prompt, True


def run_finetuned_epidemicExtractor(df, ft_model, ft_tokenizer, ft_fmt=None,
                                    max_new_tokens=FT_MAX_NEW_TOKENS,
                                    max_input_tokens=FT_MAX_INPUT_TOKENS, DEBUG=False):
    """Run jattokatarratto/EpiQwen3.5-9B on every article of `df` and return the flat extraction DataFrame.

    `ft_fmt` is the serving prompt format returned by load_finetuned_epi_model(): it selects the
    prompt and the stop token of the model (None = legacy Alpaca-prompt fine-tune).

    One row per extracted outbreak: an article reporting several outbreaks yields several rows
    (same fileid/dates/texts, increasing 'outbreak_index'), and an article for which the model
    reports nothing keeps a single row with all the extraction fields set to NaN, so that no
    article is ever silently dropped.
    """
    if ft_fmt is not None and max_new_tokens == FT_MAX_NEW_TOKENS:
        # the answer budget the model was trained and evaluated with wins over the default of the
        # pipeline (it costs nothing anyway: the generation stops on the chat stop token)
        max_new_tokens = ft_fmt.get("recommended_generation", {}).get("max_new_tokens", max_new_tokens)

    records = []
    n_articles_with_extraction = 0
    n_truncated = 0

    for _, article in tqdm(df.iterrows(), total=len(df), desc="FT extraction"):

        # the prompt of the model, with a safety cap on its length (the inference helper does
        # not truncate, so a very long full text would otherwise blow up the generation)
        prompt, truncated = ft_make_prompt(ft_tokenizer, ft_fmt, article['texts'],
                                           max_input_tokens=max_input_tokens,
                                           fileid=article['fileid'])
        if truncated:
            n_truncated += 1

        raw_response = ""
        try:
            raw_response = ft_generate_response(ft_model, ft_tokenizer, prompt,
                                                max_new_tokens=max_new_tokens, fmt=ft_fmt)
        except Exception as err:
            print(f"   Warning: extraction failed for {article['fileid']}: {err}")
            logging.error(f"Error: FT extraction failed for {article['fileid']}: {err}")

        extractions = parse_ft_extraction(raw_response)
        n_outbreaks = len(extractions)
        if extractions:
            n_articles_with_extraction += 1
        else:
            # no outbreak reported (or unparsable output): keep the article with an empty
            # extraction rather than losing the row, with n_outbreaks = 0
            extractions = [{field: np.nan for field in FT_EXTRACTION_FIELDS}]

        for idx_outbreak, extraction in enumerate(extractions):
            record = {
                'fileid': article['fileid'],
                'source_name_EIOS': article.get('source_name_EIOS', ""),
                'source_id_EIOS': article.get('source_id_EIOS', ""),
                'source_url_EIOS': article.get('source_url_EIOS', ""),
                'date_pubDate_EIOS': article.get('date_pubDate_EIOS', ""),
                'date_processedOnDate_EIOS': article.get('date_processedOnDate_EIOS', ""),
                'date_cases_IMPUTED': article.get('date_cases_IMPUTED', ""),
                'texts': article['texts'],
                'texts_length': article.get('texts_length', np.nan),
                'outbreak_index': idx_outbreak,
                'n_outbreaks': n_outbreaks,
            }
            record.update(extraction)
            record['json_original'] = raw_response
            records.append(record)

    print("\nFT extraction done: " + str(n_articles_with_extraction) + "/" + str(len(df)) +
          " articles with at least one outbreak extracted, " + str(len(records)) + " rows produced" +
          (", " + str(n_truncated) + " prompts truncated" if n_truncated else ""))

    return pd.DataFrame(records,
                        columns=FT_ARTICLE_FIELDS + FT_EXTRACTION_FIELDS + ['json_original'])


def ft_backfill_source_columns(df_cached, df_day, day_csv_filename, DEBUG=False):
    """Complete a reused day CSV with the EIOS source metadata, when it does not carry it yet.

    The extractions of a day are only computed once (they cost ~8 s of GPU per article) and read
    back from their CSV at every later run, so a day extracted before FT_SOURCE_FIELDS existed
    would keep an ekg:MediaReport without its outlet for ever. The metadata does not come from the
    model, but from the raw EIOS items of the day (`df_day`), so it can simply be joined on the
    fileid and the CSV rewritten in place: no re-extraction is needed.
    """
    missing = [column for column in FT_SOURCE_FIELDS
               if column not in df_cached.columns or df_cached[column].isna().all()]
    if not missing:
        return df_cached

    try:
        source_of_fileid = df_day.drop_duplicates(subset=['fileid']).set_index('fileid')
        for column in missing:
            if column in source_of_fileid.columns:
                df_cached[column] = df_cached['fileid'].map(source_of_fileid[column])
        # the documented column order is restored, so that the CSV of a backfilled day is not
        # different from the CSV of a day extracted from scratch
        expected = [column for column in FT_ARTICLE_FIELDS + FT_EXTRACTION_FIELDS + ['json_original']
                    if column in df_cached.columns]
        df_cached = df_cached[expected + [c for c in df_cached.columns if c not in expected]]
        df_cached.to_csv(day_csv_filename, sep=',', header=True, index=False, encoding='utf-8')
        print("   [cache] EIOS source metadata (" + ", ".join(missing) + ") added to " +
              str(day_csv_filename))
    except Exception as err:
        print(f"   Warning: could not add the EIOS source metadata to {day_csv_filename}: {err}")
        logging.error(f"Error: could not add the EIOS source metadata to {day_csv_filename}: {err}")

    return df_cached


def resolve_starting_date(days_root, fallback_starting_date=STARTING_DATE,
                          iter_ttl_basename=ITER_ONTOLOGY_BASENAME, DEBUG=False):
    """Return the "YYYYMMDD" day the day-by-day EIOS harvesting has to resume from.

    The "YYYYMMDD" subfolders of `days_root` are ranked from the most recent to the oldest and
    scanned until one holding its own ontology (`iter_ttl_basename`) is found. That day is the last
    one fully computed -- the ontology is the last artefact written for a day -- so everything is
    complete up to it and the harvesting resumes from the day right after it. A more recent day
    that has only the raw EIOS cache, and no ontology, is therefore harvested again.
    When no day holds an ontology at all (the very first run, or a working directory that only
    holds raw caches), `fallback_starting_date` is returned, i.e. the first day of the series.
    An empty string is returned when there is nothing left to harvest, i.e. when everything is
    already computed up to the last day that can be harvested: a day can only be asked to EIOS once
    it is over, so that is yesterday. The caller stops in that case: the run must never try to
    harvest today (still incomplete) or a day in the future.
    """
    day_dirs = sorted((d for d in glob.glob(os.path.join(days_root, "[0-9]" * 8))
                       if os.path.isdir(d)), reverse=True)   # most recent first

    # "YYYYMMDD" keys compare chronologically as strings
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    last_harvestable_key = (now_utc - datetime.timedelta(days=1)).strftime("%Y%m%d")

    for day_dir in day_dirs:

        day_key = os.path.basename(day_dir)

        if os.path.isfile(os.path.join(day_dir, iter_ttl_basename)):

            if day_key >= last_harvestable_key:
                # the most recent computed day is yesterday (or today, if a run has already gone
                # that far): there is no complete day left to harvest
                print("Everything is already computed up to " + str(day_key) + ": " +
                      ("today is the last computed day"
                       if day_key >= now_utc.strftime("%Y%m%d")
                       else now_utc.strftime("%d %b %Y") + " is not over yet, so it cannot be harvested") +
                      " -> nothing left to compute.")
                logging.info("Everything already computed up to " + str(day_key) +
                             ": nothing left to compute.")
                return ""

            resume_from = (datetime.datetime.strptime(day_key, "%Y%m%d") +
                           datetime.timedelta(days=1)).strftime("%Y%m%d")
            print("Last day fully computed (its " + str(iter_ttl_basename) + " is on disk) = " +
                  str(day_key) + " -> the harvesting resumes from " + str(resume_from))
            logging.info("Last day fully computed = " + str(day_key) + ": harvesting resumes from " +
                         str(resume_from))
            return resume_from

        if DEBUG:
            print("   day " + str(day_key) + " has no " + str(iter_ttl_basename) +
                  " -> not computed yet, checking the day before")

    print("No day with its own " + str(iter_ttl_basename) + " found under " + str(days_root) +
          " -> starting from the first day of the series, " + str(fallback_starting_date))
    logging.info("No computed day found under " + str(days_root) + ": starting from " +
                 str(fallback_starting_date))

    return fallback_starting_date


def write_reduced_ontology(DEBUG, source_ttl_file, reduced_ttl_file, reduced_rdf_file):
    """Write the shareable version of the overall ontology, without the source articles.

    The shareable Turtle is the complete one of `source_ttl_file`, line for line -- the same preamble,
    the same prefixes, the same schema declarations (all of them, also the ones nothing uses anymore),
    the same day sections and the same layout of the nodes -- with exactly two things taken out:
      - the ekg:MediaReport nodes, which describe the article an outbreak was extracted from and the
        media outlet that published it;
      - the foundation:hasSignal statements of the outbreak events, which are what points at those
        nodes, and their dcterms:identifier, which is the identifier of that very article.
    Nothing else is touched: the events keep their rdfs:label ("recordYYYYMMDD_i", the key they are
    published under, also carried by the shareable extractions), their description and all of their
    extracted content, and so do the diseases, the countries, the places, the impacts and the date
    structures.
    The reduction is done on the text and not on the parsed graph on purpose: a graph re-serialised by
    rdflib would be laid out differently from the complete file, while what is wanted here is the very
    same file minus the sources. The result is then parsed back, which validates the Turtle and
    produces the RDF/XML version.
    """
    try:
        with open(source_ttl_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()

        kept_lines = []
        node_lines = []        # the lines of the node being read, empty outside a node
        drop_next_blank = False
        n_reports = 0
        n_statements = 0

        for line in lines:

            if not node_lines:
                # a node starts with its subject on a line of its own, see _ekg_write_node() in
                # ONTO-RESOURCES_eKG.py; everything else here (the preamble, the comments and the
                # blank lines separating the nodes) is copied over untouched
                if line.startswith("<"):
                    node_lines = [line]
                elif drop_next_blank and not line.strip():
                    # the blank line that used to separate a dropped node from the next one
                    drop_next_blank = False
                else:
                    drop_next_blank = False
                    kept_lines.append(line)
                continue

            node_lines.append(line)
            if not line.rstrip().endswith("."):
                # the statements of a node are separated by " ;" and the last one ends with " ."
                continue

            subject_line, statements = node_lines[0], node_lines[1:]
            node_lines = []

            # the articles: the whole node of every ekg:MediaReport
            if any(statement.strip().startswith("a ") and REDUCED_ONTOLOGY_DROPPED_CLASS in statement
                   for statement in statements):
                n_reports += 1
                drop_next_blank = True
                continue

            # what points at the article: the signal link of any node, and on top of that the
            # identifier of the outbreak events, which is the identifier of their article
            dropped_predicates = list(REDUCED_ONTOLOGY_DROPPED_PREDICATES)
            if any(statement.strip().startswith("a ") and REDUCED_ONTOLOGY_EVENT_CLASS in statement
                   for statement in statements):
                dropped_predicates += REDUCED_ONTOLOGY_DROPPED_EVENT_PREDICATES

            kept = [statement for statement in statements
                    if not any(statement.strip().startswith(predicate + " ")
                               for predicate in dropped_predicates)]
            n_statements += len(statements) - len(kept)

            if not kept:
                # a node made of nothing but the dropped statement would leave a dangling subject
                drop_next_blank = True
                continue

            if len(kept) != len(statements):
                # the statement dropped may have been the last one of the node, in which case the new
                # last one has to be terminated by " ." instead of " ;"
                last = kept[-1].rstrip()
                if last.endswith(";"):
                    kept[-1] = last[:-1].rstrip() + " .\n"

            kept_lines.append(subject_line)
            kept_lines.extend(kept)

        # a node left open by a truncated file is written back as it is, so that nothing is lost
        kept_lines.extend(node_lines)

        with open(reduced_ttl_file, 'w', encoding='utf-8') as f_out:
            f_out.writelines(kept_lines)

        # parse the result back: it validates the Turtle just written and gives the RDF/XML version
        g_reduced = Graph()
        g_reduced.parse(reduced_ttl_file, format='turtle')
        if reduced_rdf_file:
            g_reduced.serialize(destination=reduced_rdf_file, format='xml')

        print("Overall ontology, shareable version (no source article): " + str(reduced_ttl_file) +
              " (" + str(len(g_reduced)) + " triples, " + str(n_reports) + " article nodes and " +
              str(n_statements) + " statements pointing at them dropped)")
        if DEBUG:
            print("Overall ontology, shareable version, RDF: " + str(reduced_rdf_file))
        logging.info(f"Reduced ontology written: {reduced_ttl_file} ({len(g_reduced)} triples, "
                     f"{n_reports} article nodes and {n_statements} statements dropped)")

        return reduced_ttl_file

    except Exception as err:
        # the reduced files are a copy of what has already been saved: their failure must not lose
        # the run, but it must be visible, since they are the files published to JRC-OpenData
        print(f"Error while writing the reduced ontology {reduced_ttl_file}: {err}")
        logging.error(f"Error while writing the reduced ontology {reduced_ttl_file}: {err}")
        return ""


def rebuild_overall_ontology(DEBUG, days_root, eKG_ontologyPREAMBLE_file_location, overall_ttl_file,
                             overall_rdf_file, iter_ttl_basename=ITER_ONTOLOGY_BASENAME,
                             reduced_ttl_file=None, reduced_rdf_file=None):
    """Rebuild the overall ontology from every per-day ontology stored on disk.

    The overall TTL/RDF are never patched incrementally: they are recomputed from scratch at every
    run, as the concatenation of all the <days_root>/YYYYMMDD/<iter_ttl_basename> in chronological
    order. This way they are always exactly the sum of what is stored on disk and they can be
    reconstructed at any moment, whatever happened in between (a crashed run, a day recomputed, a
    day removed by hand).
    Every per-day file begins with a copy of the preamble (writeOntoFile copies it before appending
    the outbreaks of the day): the preamble is written only once, at the top of the overall file,
    and is stripped from each day.
    When `reduced_ttl_file` is given, the shareable version of the result is written as well, out of
    the file just written, see write_reduced_ontology().
    """
    with open(eKG_ontologyPREAMBLE_file_location, 'r', encoding='utf-8') as f_preamble:
        preamble = f_preamble.read()

    # only the "YYYYMMDD" day folders, in chronological order (which is also the alphabetical one)
    day_ttl_files = sorted(glob.glob(os.path.join(days_root, "[0-9]" * 8, iter_ttl_basename)))

    if not day_ttl_files:
        print("Warning: no per-day ontology found under " + str(days_root) +
              " -> the overall ontology is not rebuilt.")
        logging.warning("No per-day ontology found under " + str(days_root) +
                        ": the overall ontology is not rebuilt.")
        return ""

    # the header writeOntoFile writes right after the preamble, used as a fallback to locate the
    # end of the preamble in a day generated with an older version of it
    outbreaks_header = "#    Outbreak events extracted from the EIOS articles"

    with open(overall_ttl_file, 'w', encoding='utf-8') as f_out:

        f_out.write(preamble)
        if not preamble.endswith("\n"):
            f_out.write("\n")

        for day_ttl in day_ttl_files:

            day_key = os.path.basename(os.path.dirname(day_ttl))

            with open(day_ttl, 'r', encoding='utf-8') as f_in:
                body = f_in.read()

            if body.startswith(preamble):
                body = body[len(preamble):]
            else:
                pos = body.find(outbreaks_header)
                if pos >= 0:
                    print("   Warning: " + str(day_ttl) + " was generated with a different version of"
                          " the preamble -> only its outbreaks are taken")
                    body = body[pos:]
                else:
                    print("   Warning: could not locate the preamble in " + str(day_ttl) + " -> the"
                          " file is joined as it is")

            f_out.write("\n\n#################################################################\n")
            f_out.write("#    Day " + str(day_key) + "\n")
            f_out.write("#################################################################\n\n")
            f_out.write(body)

    # parse back the file just written: it validates the Turtle and produces the RDF/XML version.
    # The same disease, place or article described by several days gives duplicate triples in the
    # concatenation: they are re-assertions of the very same statement, and they collapse here,
    # since an RDF graph is a set of triples.
    g_overall = Graph()
    g_overall.parse(overall_ttl_file, format='turtle')
    g_overall.serialize(destination=overall_rdf_file, format='xml')

    print("Overall ontology rebuilt from " + str(len(day_ttl_files)) + " day(s): " +
          str(overall_ttl_file) + " (" + str(len(g_overall)) + " triples)")
    if DEBUG:
        print("Overall ontology RDF: " + str(overall_rdf_file))
    logging.info(f"Overall ontology rebuilt from {len(day_ttl_files)} day(s): {overall_ttl_file}")

    # the shareable version: the very file just written, minus the sources of the outbreaks
    if reduced_ttl_file:
        write_reduced_ontology(DEBUG, overall_ttl_file, reduced_ttl_file, reduced_rdf_file)

    return overall_ttl_file


def rebuild_overall_extractions(DEBUG, days_root, overall_csv_file, overall_reduced_csv_file,
                                iter_csv_basename=ITER_EXTRACTIONS_BASENAME):
    """Rebuild the two overall extraction files from every per-day extraction file on disk.

    Exactly like rebuild_overall_ontology() does for the TTL/RDF: the overall CSVs are never
    appended to, they are recomputed from scratch at every run as the concatenation of all the
    <days_root>/YYYYMMDD/<iter_csv_basename> in chronological order. So they are always exactly the
    sum of what is stored on disk: duplicates are structurally impossible, a day recomputed replaces
    its own rows instead of doubling them, and a day removed by hand disappears from the overall
    files as well.
    Every row is given its RECORD_ID_FIELD, the identifier of the outbreak record it produced in the
    ontology (the rdfs:label of the ekg:OutbreakEvent), so that the extractions and the knowledge
    graph can be joined; it is empty for the rows that carry no outbreak, which have no record.
    Two files are written: the complete one, which keeps every row and every column, and the
    shareable "_red" version, which is the same frame without REDUCED_EXTRACTIONS_DROPPED_FIELDS,
    i.e. without the full text of the source article, which cannot be redistributed (copyright of the
    MedicalOfficial sources), and without what identifies the source itself, and without the rows
    that produced no record. 'texts_length' is kept: it is a length, not content, and it is published
    in the ontology as ekg:hasTextLength.
    """
    # only the "YYYYMMDD" day folders, in chronological order (which is also the alphabetical one)
    day_csv_files = sorted(glob.glob(os.path.join(days_root, "[0-9]" * 8, iter_csv_basename)))

    if not day_csv_files:
        print("Warning: no per-day extraction file found under " + str(days_root) +
              " -> the overall extraction files are not rebuilt.")
        logging.warning("No per-day extraction file found under " + str(days_root) +
                        ": the overall extraction files are not rebuilt.")
        return ""

    frames = []
    for day_csv in day_csv_files:
        try:
            # dtype=str for every day, so that the result does not depend on what pandas infers
            # from the values of a single day (a column of counts read as int for one day and as
            # str for another would otherwise make identical rows compare as different)
            df_day = pd.read_csv(day_csv, sep=',', header=0, dtype=str, encoding='utf-8')
            # the identifiers of the records of that day, computed by the ontology writer itself out
            # of the very frame its ontology was written from: the day the rows belong to is the name
            # of the folder holding them, which is the record_id_tag the day ontology was written with
            day_key = os.path.basename(os.path.dirname(day_csv))
            df_day.insert(0, RECORD_ID_FIELD, record_ids_of_extractions(df_day, day_key))
            frames.append(df_day)
        except Exception as err:
            # a broken/empty file of one day must not lose all the other days
            print(f"   Warning: could not read {day_csv}: {err}")
            logging.error(f"Could not read {day_csv}: {err}")

    if not frames:
        print("Warning: none of the " + str(len(day_csv_files)) + " per-day extraction files could be"
              " read -> the overall extraction files are not rebuilt.")
        logging.warning("None of the per-day extraction files could be read: the overall extraction"
                        " files are not rebuilt.")
        return ""

    df_overall = pd.concat(frames, ignore_index=True)

    df_overall.to_csv(overall_csv_file, sep=',', header=True, index=False, encoding='utf-8')

    dropped = [column for column in REDUCED_EXTRACTIONS_DROPPED_FIELDS
               if column in df_overall.columns]
    # only the rows that ended up as an outbreak record of the ontology: a row without record is a
    # row the model extracted neither a disease nor a country from, of no use once the article it
    # refers to is not disclosed
    df_overall_reduced = df_overall[df_overall[RECORD_ID_FIELD] != ""].drop(columns=dropped)
    df_overall_reduced.to_csv(overall_reduced_csv_file, sep=',', header=True, index=False,
                              encoding='utf-8')

    print("Overall extractions rebuilt from " + str(len(frames)) + " day(s): " +
          str(overall_csv_file) + " (" + str(len(df_overall)) + " rows, " +
          str(df_overall['fileid'].nunique() if 'fileid' in df_overall.columns else "?") +
          " articles)")
    print("Overall extractions, shareable version (no source article): " +
          str(overall_reduced_csv_file) + " (" + str(len(df_overall_reduced)) + " rows, one per"
          " outbreak record of the ontology, " + str(len(df_overall_reduced.columns)) + " columns)")
    logging.info(f"Overall extractions rebuilt from {len(frames)} day(s): {overall_csv_file} "
                 f"({len(df_overall)} rows)")

    return overall_csv_file


def main(synk_location):
    """Run one pass of the daily pipeline over all not-yet-computed days.

    ``synk_location`` is the working directory of the run: it holds one "YYYYMMDD" subfolder per
    harvested day (raw EIOS cache, per-day extractions and per-day ontology) plus the overall
    extraction and ontology files. The function resolves the resume point, harvests and
    pre-processes the EIOS articles day by day, runs the fine-tuned extraction model, builds the
    per-day and overall eKG, and rebuilds the overall extraction files (including the shareable
    "_red" versions).
    """

    print("SYS ARGV: ")
    print(sys.argv)
    print("\n")

    ## Nothing is started outside the service window: the ontology generation would fail on every
    ## entity-linking query, and there is no point in harvesting days that could not be linked
    ## anyway. The run simply does nothing and the next scheduled one picks the work up.
    #if not within_service_window():
    #    print("Not starting the pipeline: " + service_window_message() +
    #          ". Nothing is done....exiting now....\n")
    #    logging.info("Pipeline not started: " + service_window_message())
    #    return

    print(service_window_message() + "\n")

    DEBUG = True

    DATE_IMPUTATION = True # False

    # `synk_location` is the working directory of the run, given by the caller (see __main__ below,
    # or SCHEDULE-RUN_AutomatedDynamicPipeline.py when the pipeline is run by the scheduler)

    process_specify = Path(__file__).parent.resolve() / Path(synk_location)

    process_dir = str(process_specify).strip() + "/"

    # the working directory is also the root of the per-day folders, which hold everything that
    # belongs to one harvested day: the cache of the raw EIOS results
    # (<process_dir>/YYYYMMDD/{articles.json, fulltexts.json, _COMPLETE.json}) and, written further
    # down, the extractions of that day (OutputAnnotatedTexts-FT-LLM-IterUpdate.csv) and its
    # ontology (epidemicIE_IterUpdate.ttl|rdf)
    medical_official_cache_dir = process_dir

    ###########################################################################

    # Create new `pandas` methods which use `tqdm` progress
    # https://medium.com/analytics-vidhya/2-apply-method-alternatives-you-should-try-909f3f06cec6
    # (can use tqdm_gui, optional kwargs, etc.)
    tqdm.pandas()

    #############################################################################################

    # NOTE: the overall extraction file (OutputAnnotatedTexts-FT-LLM.csv) is NOT read here, and it is
    # never used as an input of the run: it is entirely rebuilt at the end from the per-day
    # extractions found on disk, see rebuild_overall_extractions().

    #############################################################################################

    start = time.time()

    # NOTE: the resume point is NOT derived from the last computed date found in
    # OutputAnnotatedTexts-FT-LLM.csv. It is read from the day folders themselves: the day after the
    # most recent one that holds its own ontology, see resolve_starting_date().
    starting_date = resolve_starting_date(medical_official_cache_dir, DEBUG=DEBUG)
    if not starting_date:
        # everything is already computed up to yesterday, the last day EIOS can be asked for:
        # nothing to harvest, and nothing to recompute (see resolve_starting_date)
        print("\n...exiting now....\n")
        return

    ###
    dataToPopulate = []
    ###

    dtnow = datetime.datetime.now(datetime.timezone.utc)
    tmsdtnow = pd.to_datetime(dtnow, errors='coerce', utc=True)

    ### GETTING THE MEDICAL OFFICIAL ARTICLES, DAY BY DAY, FROM THE RESUME POINT UNTIL TODAY

    startdate_mo = pd.to_datetime(starting_date, format="%Y%m%d", utc=True)
    enddate_mo = startdate_mo + datetime.timedelta(days=1)

    numbersOfDaysMO = tmsdtnow - startdate_mo
    print("MEDICAL OFFICIAL - starting date = " + str(starting_date))
    print("MEDICAL OFFICIAL - Number of days required = " + str(numbersOfDaysMO.days) + " - Period: " + startdate_mo.strftime("%d %b %Y") + " - " + tmsdtnow.strftime("%d %b %Y"))
    print("MEDICAL OFFICIAL - per-day cache = " + str(medical_official_cache_dir))
    print("\n")

    if startdate_mo >= tmsdtnow:
        # every day up to today is already computed: there is nothing to harvest, and the EIOS
        # client is not even created
        print("Everything is already computed up to " + tmsdtnow.strftime("%d %b %Y") +
              ": nothing to harvest....exiting now....\n")
        logging.info("Everything already computed up to " + tmsdtnow.strftime("%Y-%m-%d") +
                     ": nothing to harvest.")
        return

    #EIOS_client = get_EIOS_client()
    EIOS_client = get_EIOS_test_client()  #using the test environment because it has the fullText

    # #trialdate = enddate_mo + datetime.timedelta(days=10)
    # #trialdate = enddate_mo + datetime.timedelta(days=2)
    #trialdate = pd.to_datetime("20230503", format="%Y%m%d", utc=True)
    #while startdate_mo < trialdate:  # for demo purpose, we do only few days
    while enddate_mo < tmsdtnow:
        
        day_key = startdate_mo.strftime("%Y%m%d")
        # the daily window is [day 00:00, day 23:59]; adapt_date_format() appends ":00.000Z"
        startdate_mo_ts_string = startdate_mo.strftime("%Y-%m-%dT00:00")
        enddate_mo_ts_string = startdate_mo.strftime("%Y-%m-%dT23:59")

        if DEBUG:
            print("\nGetting MEDICAL OFFICIAL full-texts via EIOS - " + startdate_mo.strftime("%d %b %Y") + " (day " + day_key + ")")

        medicalArticlesJson, medicalFullTexts, from_cache = getMedicalOfficialArticlesCached(
            day_key, startdate_mo_ts_string, enddate_mo_ts_string, EIOS_client,
            cache_root=medical_official_cache_dir, DEBUG=DEBUG)

        ### INPUT DATAFRAME : ###

        if medicalArticlesJson and medicalArticlesJson.get("count", 0) > 0:

            for idYY, YY in enumerate(medicalArticlesJson.get("result") or []):

                a=""
                item_id = YY.get('id', '')
                if YY.get('eiosUrl'):
                    a = MEDICAL_OFFICIAL_FILEID_PREFIX + YY['eiosUrl']
                elif item_id:
                    a = MEDICAL_OFFICIAL_FILEID_PREFIX + str(item_id)
                elif YY.get('rssItemId'):
                    a = MEDICAL_OFFICIAL_FILEID_PREFIX + YY['rssItemId']
                elif YY.get('link'):
                    a = MEDICAL_OFFICIAL_FILEID_PREFIX + YY['link']

                # the text comes from the per-day cache (already fetched via get_item_details)
                b = medicalFullTexts.get(str(item_id), "") if item_id else ""

                # the media outlet the item comes from, as the api returns it:
                #   "source": {"name": "Winston-Salem Journal", "id": "journalnow",
                #              "url": "http://www.journalnow.com/", ...}
                # only name/id/url are kept, and any of them may be missing on a given item
                item_source = YY.get('source') if isinstance(YY.get('source'), dict) else {}
                s_name = str(item_source.get('name') or "").strip()
                s_id = str(item_source.get('id') or "").strip()
                s_url = str(item_source.get('url') or "").strip()

                # the two dates as they come from the EIOS api call, kept both as columns
                d_pub = ""
                if YY.get('pubDate') and len(YY['pubDate']) > 0:
                    d_pub = isoparse(YY['pubDate']).strftime("%Y-%m-%d")  # '2023-06-23T22:30:00.0000000Z'
                d_proc = ""
                if YY.get('processedOnDate') and len(YY['processedOnDate']) > 0:
                    d_proc = isoparse(YY['processedOnDate']).strftime("%Y-%m-%d")

                # the imputed date: the publication date when available, the processing date otherwise
                c=""
                if DATE_IMPUTATION == True:
                    c = d_pub or d_proc

                if a and b and c:
                    # `day_key` (the "YYYYMMDD" of the harvested day) travels with the article: the
                    # extractions are stored back in the folder of that same day, see further down
                    dataToPopulate.append([day_key, a, s_name, s_id, s_url, d_pub, d_proc, b, c])


        startdate_mo = startdate_mo + datetime.timedelta(days=1)
        enddate_mo = startdate_mo + datetime.timedelta(days=1)
        if not from_cache:
            time.sleep(2)  # be gentle with the API only when we actually called it

        ## Stop harvesting when the service window closes: the days already harvested are extracted
        ## and turned into their ontology below, and the next scheduled run resumes from the first day
        ## that has no ontology yet (see resolve_starting_date()).
        #if not within_service_window():
        #    print("\nBreaking Medical Official loop: " + service_window_message() +
        #          ". Will resume at the next scheduled run.")
        #    logging.info("Medical Official loop interrupted: " + service_window_message())
        #    break

    # end for cycle Medical Official articles

    ############################################################################################
    ### The former ProMED-only and DONs-only harvesting loops have been replaced by the single
    ### Medical Official loop above. getPromedArticles()/getDONSArticles()/getGPHINArticles()
    ### are kept defined at the top of this file, but they are not called anymore.
    ############################################################################################


    #
    endGettingFromEIOS = time.time()
    hours, rem = divmod(endGettingFromEIOS - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nComputational Time for getting the full-text articles from EIOS... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    #


    if len(dataToPopulate)<=0:        
        print("\nNo data to insert for the period....exiting now....")
        #sys.exit()
        return
    

    df = pd.DataFrame(dataToPopulate, columns=['day_key', 'fileid'] + FT_SOURCE_FIELDS +
                                              ['date_pubDate_EIOS', 'date_processedOnDate_EIOS',
                                               'texts', 'date_cases_IMPUTED'])


    df['texts'] = df['texts'].replace(r'\(\'\\n\\n', ' ', regex=True)
    df['texts'] = df['texts'].replace(r'\(\"\\n\\n', ' ', regex=True)
    df['texts'] = df['texts'].replace(r'\\n\\n\',\)', ' ', regex=True)
    df['texts'] = df['texts'].replace(r'\\n\\n\",\)', ' ', regex=True)

    while df['texts'].str.contains(r'##\n').any():
        df['texts'] = df['texts'].str.replace(r"##\n", '. ', regex=True)
    while df['texts'].str.contains('###').any():
        df['texts'] = df['texts'].str.replace("###", ' ')
    while df['texts'].str.contains('##').any():
        df['texts'] = df['texts'].str.replace("##", ' ')
    while df['texts'].str.contains(' # ').any():
        df['texts'] = df['texts'].str.replace(" # ", ' ')
    while df['texts'].str.contains('--').any():
        df['texts'] = df['texts'].str.replace("--", '-')
    while df['texts'].str.contains(r'\\\\-').any():
        df['texts'] = df['texts'].str.replace(r"\\\\-", '.', regex=True)
    while df['texts'].str.contains(r'\*\*\n').any():
        df['texts'] = df['texts'].str.replace(r"\*\*\n", '. ', regex=True)
    while df['texts'].str.contains(r'\*\*\*').any():
        df['texts'] = df['texts'].str.replace(r"\*\*\*", ' ', regex=True)
    while df['texts'].str.contains(r'\*\*').any():
        df['texts'] = df['texts'].str.replace(r"\*\*", ' ', regex=True)
    while df['texts'].str.contains(r' \* ').any():
        df['texts'] = df['texts'].str.replace(r" \* ", ' ', regex=True)
    while df['texts'].str.contains(r'is a program of the\n\nInternational Society for Infectious Diseases').any():
        df['texts'] = df['texts'].replace(r'is a program of the\n\nInternational Society for Infectious Diseases',
                                          'is a program of the International Society for Infectious Diseases',
                                          regex=True)

    # while df['texts'].str.contains(r'\n\n').any():
    #     df['texts'] = df['texts'].replace(r'\n\n', '. ', regex=True)
    # while df['texts'].str.contains(r'\n').any():
    #     df['texts'] = df['texts'].replace(r'\n', ' ', regex=True)

    while df['texts'].str.contains(r' \*\.').any():
        df['texts'] = df['texts'].str.replace(r' \*\.', ' .', regex=True)
    while df['texts'].str.contains('  ').any():
        df['texts'] = df['texts'].str.replace("  ", ' ')
    while df['texts'].str.contains(r'\.\.').any():
        df['texts'] = df['texts'].str.replace(r'\.\.', '.', regex=True)
    while df['texts'].str.contains(r'\. \.').any():
        df['texts'] = df['texts'].str.replace(r'\. \.', '.', regex=True)

    df['texts'] = df['texts'].replace(r'\(\"\.', ' ', regex=True)
    df['texts'] = df['texts'].replace(r'\(\'\.', ' ', regex=True)
    df['texts'] = df['texts'].replace(r'\",\)', ' ', regex=True)
    df['texts'] = df['texts'].replace(r'\',\)', ' ', regex=True)

    df['texts'] = df['texts'].astype(str).str.strip()

    df['texts_length'] = df['texts'].astype(str).str.strip().str.len()

    ########################## END ALL PREPROCESSING


    print("\n...RUNNING the Epidemic Extractor using the FINE-TUNED LLM " + str(FT_MODEL_HF_ID) + "...")
    print("\n")

    ############################################################################################
    ### The former 3-LLM ensemble (run_epidemicExtractor over MODELS), the vocabulary/embedding
    ### update (CalculateEmbeddingsVocabs / attachToDicts / attachCentroidsToDict) and the
    ### majority voting (run_EnsembleComputation) have all been replaced by a single pass of the
    ### fine-tuned model jattokatarratto/EpiQwen3.5-9B, which returns the granular extraction directly.
    ############################################################################################

    # The extractions are stored next to the raw EIOS results of the day the articles come from,
    # i.e. in <cache_root>/YYYYMMDD/ together with articles.json / fulltexts.json / _COMPLETE.json,
    # and NOT in a folder named after the day of the run: a given EIOS day always yields the same
    # extractions, so they belong to that day and any later run can reuse them.
    # They are by far the most expensive step of the pipeline (~10 s per article on the GPU), so
    # each day whose cached CSV covers exactly the articles just harvested for it is read back, and
    # the fine-tuned model is run (and loaded, lazily and only once) only for the remaining days.
    day_extractions = {}   # day_key -> (folder of the day, CSV of the day, extractions of the day)
    ft_model, ft_tokenizer, ft_fmt = None, None, None

    for day_key, df_day in df.groupby('day_key', sort=True):

        day_key = str(day_key)
        dir_path_day = os.path.join(medical_official_cache_dir, day_key) + "/"
        os.makedirs(dir_path_day, exist_ok=True)
        day_csv_filename = dir_path_day + ITER_EXTRACTIONS_BASENAME

        df_ft_day = None
        if FT_REUSE_ITER_EXTRACTIONS and os.path.isfile(day_csv_filename):
            try:
                df_cached = pd.read_csv(day_csv_filename, sep=',', header=0, dtype=str, encoding='utf-8')
                if set(df_day['fileid'].dropna()) == set(df_cached['fileid'].dropna()):
                    df_ft_day = ft_backfill_source_columns(df_cached, df_day, day_csv_filename, DEBUG=DEBUG)
                    print("   [cache] extractions of day " + day_key + " read from " + str(day_csv_filename) +
                          " (" + str(len(df_ft_day)) + " rows, " + str(df_ft_day['fileid'].nunique()) +
                          " articles) -> the fine-tuned model is NOT run again")
                else:
                    print("   [cache] " + str(day_csv_filename) + " does not cover the same articles"
                          " as the ones just harvested for day " + day_key + " -> recomputing the extractions")
            except Exception as err:
                print(f"   Warning: could not reuse {day_csv_filename}: {err}")

        if df_ft_day is None:

            if ft_model is None:
                ft_model, ft_tokenizer, ft_fmt = load_finetuned_epi_model()

            print("\n   day " + day_key + ": extracting from " + str(len(df_day)) + " articles")

            df_ft_day = run_finetuned_epidemicExtractor(df_day, ft_model, ft_tokenizer, ft_fmt=ft_fmt,
                                                        max_new_tokens=FT_MAX_NEW_TOKENS,
                                                        max_input_tokens=FT_MAX_INPUT_TOKENS,
                                                        DEBUG=DEBUG)

            # save the extractions of the day BEFORE anything else is done with them, so that they
            # survive a failure of the ontology generation that follows
            df_ft_day.to_csv(day_csv_filename, sep=',', header=True, index=False, encoding='utf-8')
            print("Extractions of day " + day_key + " saved in " + str(day_csv_filename))

        day_extractions[day_key] = (dir_path_day, day_csv_filename, df_ft_day)

    if ft_model is not None:
        # free the GPU memory: the model is not needed anymore for the rest of the pipeline
        try:
            del ft_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as err:
            print(f"   Warning: could not release the fine-tuned model: {err}")

    #
    endRunningEpidemicExtractor = time.time()
    hours, rem = divmod(endRunningEpidemicExtractor - endGettingFromEIOS, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nComputational Time for running the Epidemic Extractor using the fine-tuned LLM... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    #
    # kept for the timing reports further down, which used to be relative to the dictionary step
    endRunningDictPopulation = endRunningEpidemicExtractor

    # summary of what this iteration produced. Nothing downstream works on the extractions of the
    # iteration as a single frame anymore: the ontology is generated one day at a time from the CSV
    # of the day, and the overall extraction files are rebuilt from all the per-day CSVs on disk.
    print("\nExtractions of this iteration: " +
          str(sum(len(df_day) for _, _, df_day in day_extractions.values())) + " rows over " +
          str(len(day_extractions)) + " day(s) " + str(sorted(day_extractions)))

    ######### ONTOLOGY GENERATION for the iterative update

    print("\n...GENERATING ONTOLOGY for the iterative update...")
    print("\n")

    startRunningOntology = time.time()

    try:
        # the preamble is a source file of the project, taken from the script directory (and NOT
        # from process_dir), so that it does not depend on the working directory of the run
        eKG_ontologyPREAMBLE_file_location = str(Path(__file__).parent.resolve() / "epidemicIE_PREAMBLE.ttl")

        # one ontology per harvested day, written in the folder of the day next to its extractions.
        # The day is also the tag of the record identifiers (record<YYYYMMDD>_<index>), so that the
        # records of two days do not collide once all the days are joined in the overall ontology.
        # This is the step that needs the entity-linking services (Virtuoso SPARQL / GeoNames), so
        # each day is only started inside the service window: the extractions of the days left out
        # are already saved in their own folder, and they get their ontology at the next run.
        days_with_new_ontology = []

        for day_key in sorted(day_extractions):

            # the day is only started if the window stays open long enough for its entity linking
            if not within_service_window(margin_minutes=SERVICE_WINDOW_ONTOLOGY_MARGIN_MINUTES):
                margin_msg = service_window_message(
                    margin_minutes=SERVICE_WINDOW_ONTOLOGY_MARGIN_MINUTES)
                print("\nOntology generation interrupted: " + margin_msg + ". " +
                      str(len(days_with_new_ontology)) + " day(s) done, the remaining ones (" +
                      str([d for d in sorted(day_extractions) if d not in days_with_new_ontology]) +
                      ") keep their extractions and are computed at the next scheduled run.")
                logging.info("Ontology generation interrupted: " + margin_msg +
                             f" ({len(days_with_new_ontology)} day(s) done)")
                break

            dir_path_day, day_csv_filename, _ = day_extractions[day_key]
            iter_ttl_filename = dir_path_day + ITER_ONTOLOGY_BASENAME

            try:
                iter_ttl_result = writeOntoFile(
                    DEBUG,
                    eKG_ontologyPREAMBLE_file_location,
                    day_csv_filename,
                    PerformEntityLinking=True,
                    USE_CACHE=True,
                    output_ttl_path=iter_ttl_filename,
                    KG_restriction=KG_restriction,
                    record_id_tag=day_key
                )
            except Exception as err:
                # a failure on one day must not lose the days that follow
                print(f"Error during ontology generation of day {day_key}: {err}")
                logging.error(f"Error during ontology generation of day {day_key}: {err}")
                continue

            if not iter_ttl_result:
                print("Warning: ontology generation of day " + day_key + " returned empty result.")
                logging.warning("Ontology generation of day " + day_key + " returned empty result.")
                continue

            # Convert the TTL of the day to RDF
            iter_rdf_filename = iter_ttl_filename.replace(".ttl", ".rdf")
            convert_turtle_to_rdf(DEBUG, iter_ttl_result, iter_rdf_filename)
            days_with_new_ontology.append(day_key)
            logging.info(f"Ontology of day {day_key} generated: {iter_ttl_result}")

        # The overall ontology is rebuilt from scratch, joining ALL the per-day ontologies present
        # under process_dir (not only the days of this run): see rebuild_overall_ontology(). It needs
        # no service, only the files on disk.
        # ... together with its shareable version (epidemicIE_red.ttl / .rdf), which carries the same
        # outbreak events without the articles they were extracted from, see write_reduced_ontology()
        rebuild_overall_ontology(DEBUG, process_dir, eKG_ontologyPREAMBLE_file_location,
                                 process_dir + "epidemicIE.ttl", process_dir + "epidemicIE.rdf",
                                 reduced_ttl_file=process_dir + "epidemicIE_red.ttl",
                                 reduced_rdf_file=process_dir + "epidemicIE_red.rdf")

    except Exception as e:
        print(f"Error during ontology generation: {e}")
        logging.error(f"Error during ontology generation: {e}")

    #
    endRunningOntology = time.time()
    hours, rem = divmod(endRunningOntology - startRunningOntology, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nComputational Time for Ontology Generation... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    #

    ####rebuild and save the overall extraction files

    ######## The two overall extraction files are rebuilt from scratch, joining ALL the per-day
    ######## extractions present under process_dir (not only the days of this run), exactly as the
    ######## overall ontology is: see rebuild_overall_extractions(). They are NOT obtained by
    ######## appending this iteration to the previous overall file, so no duplicate row can appear
    ######## when a day already in the overall file is recomputed.
    ######## The second file is the shareable version, without the full text of the source article
    ######## (copyright of the MedicalOfficial sources).
    ######## The former ProMED/DONs-specific handling (fileids collapsed to 'P'/'D', and the
    ######## separate _whoDons output) is gone: the harvesting is MedicalOfficial-only now.

    rebuild_overall_extractions(DEBUG, process_dir,
                                process_dir + "OutputAnnotatedTexts-FT-LLM.csv",
                                process_dir + "OutputAnnotatedTexts-FT-LLM_red.csv")

    ########




    #
    endRunningEnsemble = time.time()
    hours, rem = divmod(endRunningEnsemble - endRunningDictPopulation, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nComputational Time for running the Ensemble pipeline... {:0>2}:{:0>2}:{:05.2f}\n".format(
        int(hours), int(minutes), seconds))
    #


    ######### ######### ######### ######### #########

    #
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nOverall Computational Time... {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    #

    print("\nEnd Computations")





if __name__ == "__main__":

    # Working directory of the run: it holds one "YYYYMMDD" subfolder per harvested day (the cache
    # of the raw EIOS results, the extractions of that day and its ontology) plus the overall
    # OutputAnnotatedTexts-FT-LLM*.csv / epidemicIE.ttl|rdf files. Configure it with the
    # EPIDEMIC_IE_WORKDIR environment variable or pass it as the first command-line argument;
    # otherwise it defaults to ./data/ relative to the project directory.
    synk_location = os.environ.get("EPIDEMIC_IE_WORKDIR",
                                   str(Path(__file__).parent.resolve() / "data") + os.sep)
    if (("/jupyter/" in str(sys.argv)) == False) and len(sys.argv) > 1:
        synk_location = sys.argv[1]
    print("synk_location = " + str(synk_location) + "\n")

    main(synk_location)
