# API key files

Several components of the pipeline read their credentials at run time from plain-text `*.key`
files located in the **project root** (the directory that contains the Python modules), not in this
`keys/` folder. Each file must contain only the secret value, on a single line, with no quotes.

All `*.key` files are gitignored and must never be committed. This folder ships only `*.key.example`
templates documenting what each file is for. Create the real files yourself.

| Key file                           | Used by                     | What it is / where to get it |
|-------------------------------------|-----------------------------|------------------------------|
| `GEONAMES-API.key`                  | `ONTO-RESOURCES_eKG.py`     | GeoNames account username/token used for place entity linking. Register a free account at https://www.geonames.org and enable the web services. |
| `NCBO-BioPortal.key`                | `ONTO-RESOURCES_eKG.py`     | NCBO BioPortal API key for disease entity linking via BioPortal. Get it from https://bioportal.bioontology.org (your account "Account settings"). |
| `VIRTUOSO-dba.key`                  | `ONTO-RESOURCES_eKG.py`, `virtuosoQueryRest.py` | Password of the Virtuoso `dba` user for the SPARQL endpoint / WebDAV upload used in entity linking and graph publication. Only needed if you run your own Virtuoso triple store. |
| `GPTJRC-APItoken.key`               | `llmquery.py`               | Bearer token for an OpenAI-compatible LLM gateway (optional; only for the remote-API backends). Also set `GPTJRC_API_BASE`. |
| `GPTEC-APItoken.key`                | `llmquery.py`               | Bearer token for a second OpenAI-compatible LLM gateway (optional). Also set `GPTEC_API_BASE`. |
| `OpenAI-APItoken.key`               | `llmquery.py`               | OpenAI API key (optional; only if you use the OpenAI backend). Get it from https://platform.openai.com. |
| `DGLC-APItoken.key`                 | `llmquery.py`               | Token for a legacy inference backend (optional; the related code paths are disabled by default). |
| `JRC-ResearchData-Token.key`        | `jrc_research_data.py`      | Personal access token of the owner of the research-data record, used to publish new versions. Create it in the research-data repository under "Applications -> Personal access tokens". |
| `JRC-ResearchData-ShareLink.key`    | `jrc_research_data.py`      | Token of a secret share link of the record (alternative to the personal token). |

> Note on the Virtuoso key filename: the pipeline (`writeOntoFile`) reads `VIRTUOSO-dba.key`. The
> standalone demo at the bottom of `ONTO-RESOURCES_eKG.py` reads `VIRTUOSO.key`; create whichever
> the entry point you run expects (or both).

The LLM extraction itself runs **locally** with the open-weight fine-tuned model and needs no API
token (see the main README). The remote-LLM backends in `llmquery.py` are optional.
