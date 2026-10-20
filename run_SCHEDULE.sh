#!/bin/bash
#
# Launcher of the EIOS epidemic information-extraction scheduler.
#
# The scheduler (SCHEDULE-RUN_AutomatedDynamicPipeline.py) runs for ever: it starts the pipeline
# every working day at the opening of the service window. Run it inside a long-lived terminal
# session so that it survives an SSH logout, e.g. with screen:
#
#     cd /path/to/epidemic-ie-pipeline
#     screen -L                    # logs everything to ./screenlog.0
#     ./run_SCHEDULE.sh
#     Ctrl-A d                     # detach; reattach later with "screen -r"
#
# The working directory must be the project directory: the API keys are opened by bare name
# (e.g. GEONAMES-API.key) and the joblib SPARQL cache lives in the relative "cached/" folder, so
# running from anywhere else would ignore the warm cache and fail to authenticate. The line below
# changes to the directory of this script, so it can be launched from anywhere.
cd "$(dirname "$0")" || exit 1

# Python interpreter to use. Override by exporting PYTHON before running, e.g.
#   export PYTHON="$(conda run -n epidemic-ie which python)"
# Defaults to the "python" on PATH (activate your environment first).
PYTHON="${PYTHON:-python}"

# Optional: set the working directory of the pipeline (one "YYYYMMDD" folder per day plus the
# overall output files). Defaults to ./data/ relative to this script when left unset.
# export EPIDEMIC_IE_WORKDIR="/path/to/data"

# "-u": unbuffered output, so the terminal/screen log stays in sync with the pipeline. "exec": the
# python process replaces this shell, so Ctrl-C and any signal reach the pipeline directly.
exec "$PYTHON" -u SCHEDULE-RUN_AutomatedDynamicPipeline.py
