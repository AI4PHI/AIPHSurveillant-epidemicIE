# AI Public Health Surveillant - Epidemic Information-Extraction Pipeline

> **License note:** this repository is released under the **EUPL-1.2** (see [`LICENSE`](LICENSE)).
> Please confirm that this is the license intended for publication before distributing.

A daily-updated pipeline that turns WHO/EIOS outbreak reports into a structured, machine-readable
**epidemiological knowledge graph (eKG)**. Each working day the pipeline acquires the day's outbreak
items from the WHO Epidemic Intelligence from Open Sources (EIOS) platform, extracts structured
epidemiological information with a **fine-tuned open-weight large language model running locally**,
and builds an RDF knowledge graph modelled on the European Crisis Management Ontology (ECMO), with
entity linking of diseases and places to external resources. The graph is published daily to the
JRC Data Catalogue.

This repository accompanies the paper *From Outbreak Reports to a Daily-Updated Epidemiological
Knowledge Graph* (Consoli et al.; see [Citation](#citation)). The source code is hosted at
<https://github.com/AI4PHI/AIPHSurveillant-epidemicIE>.

## Why local inference

The extraction model is an open-weight model (a LoRA fine-tune of Qwen3.5-9B, released as
**EpiQwen3.5-9B**) served **locally** on modest, reasonable computing power. This design choice is
central to the pipeline:

- **Robustness and reproducibility.** A locally hosted open-weight model is not subject to the
  deprecation or withdrawal of larger proprietary generative models, so a continuously updated
  pipeline cannot be broken by an external provider retiring an API.
- **Speed and reliability for a daily pipeline.** Local inference avoids round-trips over the
  internet to cloud-hosted models, which keeps the daily run fast and removes a dependency on
  external services that could become unavailable.

## Architecture / overview

The daily job runs these stages in order:

1. **Scheduler** (`SCHEDULE-RUN_AutomatedDynamicPipeline.py`) — a long-running process that starts
   the daily job on working days inside a service window, and optionally publishes the shareable
   outputs as a new version of a research-data record.
2. **EIOS acquisition** (`eios_client.py`, called from `llm_extraction_AutomatedDynamicPipeline.py`)
   — queries the EIOS API day by day, retrieving the English items from medical and official
   sources.
3. **Pre-processing** — normalisation and de-duplication of the raw articles.
4. **LLM extraction** (`llm_extraction_AutomatedDynamicPipeline.py`, `serving_format.py`) — runs the
   fine-tuned model locally to extract structured outbreak fields from each report.
5. **ECMO semantic modelling** (`ONTO-RESOURCES_eKG.py`) — assembles a per-day RDF graph on top of
   the ECMO-based ontology preamble (`epidemicIE_PREAMBLE.ttl`).
6. **Entity linking** (`ONTO-RESOURCES_eKG.py`, `virtuosoQueryRest.py`) — links places to GeoNames
   and diseases to biomedical ontologies (via a Virtuoso SPARQL endpoint or NCBO BioPortal).
7. **Assembly and publication** (`llm_extraction_AutomatedDynamicPipeline.py`,
   `jrc_research_data.py`) — rebuilds the overall extraction files and the overall ontology
   (including shareable "_red" versions) and publishes them.

Each harvested day is cached in its own `YYYYMMDD` folder under the working directory, so a run
resumes from the first day that has not been fully computed yet.

## Repository structure

| Path | Description |
|------|-------------|
| `SCHEDULE-RUN_AutomatedDynamicPipeline.py` | Scheduler; starts the daily job on working days and triggers publication. |
| `llm_extraction_AutomatedDynamicPipeline.py` | Core daily pipeline: resume logic, EIOS acquisition, pre-processing, local LLM extraction, per-day and overall graph assembly. |
| `causal_LLM_inference-Run-EPIextraction.py` | Fine-tuned-model inference helpers (prompt building, local model loading, generation) used by the pipeline; also runnable standalone to extract from a single text. |
| `ONTO-RESOURCES_eKG.py` | ECMO-based semantic modelling and entity linking (GeoNames, Virtuoso SPARQL, BioPortal). |
| `llmquery.py` | Helpers for OpenAI-compatible LLM APIs (optional remote backends; the main extraction is local). |
| `eios_client.py` | EIOS API client (OAuth2 client-credentials auth, article retrieval). |
| `serving_format.py` | Prompt/chat-template and generation settings for the fine-tuned model. |
| `virtuosoQueryRest.py` | REST client for the Virtuoso SPARQL endpoint, with on-disk query cache. |
| `jrc_research_data.py` | Publishes outputs as new versions of a research-data (InvenioRDM) record. |
| `epidemicIE_PREAMBLE.ttl` | Ontology preamble (ECMO-based) read at run time; source file of the project. |
| `config.example.py` | Template for `config.py` (EIOS credentials). Copy and fill in. |
| `keys/` | `*.key.example` templates and documentation for the API-key files. |
| `tests/` | Standalone scripts to exercise the EIOS API client and extraction queries. |
| `requirements.txt`, `environment.yml` | Dependency specifications (pip and conda). |
| `run_SCHEDULE.sh` | Launcher for the scheduler inside a long-lived terminal session. |
| `LICENSE`, `CITATION.cff` | Licence (EUPL-1.2) and citation metadata. |

## Installation

With conda:

```bash
conda env create -f environment.yml
conda activate epidemic-ie
```

Or with pip (Python 3.12 recommended):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch` is pinned to a CUDA 13.0 build that is not on PyPI. On a different platform, drop the
`+cu130` local version and install the build matching your local CUDA (or the CPU build), e.g.:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

A CUDA-capable GPU is recommended for local LLM inference.

## Configuration

Credentials are **not** stored in the repository. Create them locally; they are gitignored.

1. **EIOS credentials.** Copy the template and fill in your values:

   ```bash
   cp config.example.py config.py
   ```

   You need EIOS API access (Azure AD OAuth2 client credentials); request it from your EIOS
   administrator. See the EIOS API documentation:
   <https://eios.who.int/portal/api/swagger/index.html>.

2. **API key files.** Several components read secrets from plain-text `*.key` files in the project
   root (one value per file). The `keys/` folder ships `*.key.example` templates and a
   [`keys/README.md`](keys/README.md) explaining each one (GeoNames, NCBO BioPortal, Virtuoso, the
   research-data tokens, and optional LLM-gateway tokens). Create the real files yourself.

3. **Environment variables (optional).**
   - `EPIDEMIC_IE_WORKDIR` — working directory of the run (defaults to `./data/`).
   - `EPIDEMIC_IE_COPY_DIR` — optional secondary directory to copy shareable outputs to.
   - `GPTJRC_API_BASE`, `GPTEC_API_BASE` — base URLs of OpenAI-compatible LLM gateways, if you use
     the optional remote backends in `llmquery.py`. These were originally institution-internal
     gateways; replace them with your own API base.

## Usage

Run the scheduler (keeps running, starts the job every working day):

```bash
./run_SCHEDULE.sh
```

Run a single pass directly (useful for a one-off or a catch-up run):

```bash
python llm_extraction_AutomatedDynamicPipeline.py /path/to/working/dir
# or, using the environment variable:
EPIDEMIC_IE_WORKDIR=/path/to/working/dir python llm_extraction_AutomatedDynamicPipeline.py
```

To exercise the EIOS API client (requires a valid `config.py`):

```bash
python tests/eios_api_test.py
python tests/eios_api_test_extractions.py
```

## Data and model availability

- **Fine-tuned extraction model (EpiQwen3.5-9B):**
  <https://huggingface.co/jattokatarratto/EpiQwen3.5-9B>
- **Epidemiological knowledge graph (eKG) in the JRC Data Catalogue:**
  <https://data.jrc.ec.europa.eu/dataset/1ff56505-cd10-4cbf-9bd3-fdb500fe43db>
- **Public SPARQL endpoint:**
  <https://api-vast.jrc.service.ec.europa.eu/sparql>
- **Interactive demo:**
  <https://jrc-ai-epidemicie.hf.space>
- **European Crisis Management Ontology (ECMO):** the ontology this eKG is modelled on; see the
  JRC/EU resources and the `data.europa.eu/h8v/ecmo/...` namespaces referenced in
  `epidemicIE_PREAMBLE.ttl`.

## Citation

This repository accompanies:

> Consoli, S., Bertolini, L., Biazzo, I., Muraro, D., Schuh, L., & Stilianakis, N. I. (2026).
> *From Outbreak Reports to a Daily-Updated Epidemiological Knowledge Graph.*
> European Commission, Joint Research Centre (JRC).
> **Submitted to Frontiers in Public Health** (DOI to be assigned upon publication).

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

This project is licensed under the **European Union Public Licence (EUPL) v. 1.2**. See
[`LICENSE`](LICENSE) for the full text. Copyright (c) 2026 European Commission, Joint Research
Centre (JRC).
