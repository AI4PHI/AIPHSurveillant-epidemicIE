"""Scheduler for the daily epidemiological information-extraction pipeline.

This long-running process starts the daily job (``main()`` in
``llm_extraction_AutomatedDynamicPipeline``) every working day at the opening of the service window
and, optionally, publishes the shareable outputs as a new version of a research-data record
(``jrc_research_data``). It is meant to be kept alive inside a ``screen``/``tmux`` session; see
``run_SCHEDULE.sh``.

The ``schedule`` library is used instead of cron so that the whole thing is self-contained and does
not depend on the host crontab or on network file systems being visible to cron.
"""

import os
import time
import datetime
import sys

import schedule
    # https://schedule.readthedocs.io/en/stable/
    # https://schedule.readthedocs.io/en/stable/examples.html

import logging


import shutil

# the service window (working days, 08:15-19:00 local time) is defined once, in the pipeline module:
# the services the pipeline needs (Virtuoso SPARQL endpoint of the entity linking above all) are only
# up in that window, so both the pipeline itself and this scheduler stick to it.
from llm_extraction_AutomatedDynamicPipeline import (main, within_service_window,
                                                    service_window_message,
                                                    SERVICE_WINDOW_START, SERVICE_WINDOW_END)



# the working directory of the pipeline: it holds one "YYYYMMDD" subfolder per harvested day (the
# cache of the raw EIOS results, the extractions of that day and its ontology) and the overall
# OutputAnnotatedTexts-FT-LLM*.csv / epidemicIE.ttl|rdf files.
# It must be the same location as the default of llm_extraction_AutomatedDynamicPipeline.main().
# Configure it with the EPIDEMIC_IE_WORKDIR environment variable; otherwise it defaults to ./data/
# relative to this file.
synk_location = os.environ.get(
    "EPIDEMIC_IE_WORKDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data") + os.sep)

print("synk_location = " + str(synk_location) + "\n")

# Optional secondary location the shareable output files can be copied to (e.g. a shared drive).
# Configure it with the EPIDEMIC_IE_COPY_DIR environment variable; left unset by default, in which
# case the copy step below is skipped and only the research-data publication is used.
JRCOpenData_copy_location = os.environ.get("EPIDEMIC_IE_COPY_DIR", "")
print("JRCOpenData_copy_location = " + str(JRCOpenData_copy_location) + "\n")

# Set up logging
# log_filename = synk_location + "job_schedule.log"
log_filename = os.path.join(synk_location, "job_schedule.log")
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def copy_to_JRCOpenData(basename):
    """Copy one output file of the pipeline to the secondary copy location, reporting what happened.

    shutil.copy2 is used instead of os.system('cp'): os.system does not raise on failure, so a
    missing source file or an unavailable destination (e.g. a network share not mounted) used to be
    reported as "Copy done!".
    """
    source_file = os.path.join(synk_location, basename)
    destination_file = os.path.join(JRCOpenData_copy_location, basename)
    print("source_file = " + source_file)
    print("destination_file = " + destination_file)
    logging.info("source_file = " + source_file)
    logging.info("destination_file = " + destination_file)
    try:
        shutil.copy2(source_file, destination_file)
        print("Copy done!")
        logging.info('Copy done!')
        return True
    except FileNotFoundError as e:
        # either the pipeline did not produce the file, or the destination folder is not there
        print(f"Error: file not found while copying '{source_file}' -> '{destination_file}': {e}")
        logging.error(f"Error: file not found while copying '{source_file}' -> '{destination_file}': {e}")
    except PermissionError as e:
        print(f"Error: permission denied while copying '{source_file}' -> '{destination_file}': {e}")
        logging.error(f"Error: permission denied while copying '{source_file}' -> '{destination_file}': {e}")
    except Exception as e:
        print(f"Error: an error occurred while copying '{source_file}' -> '{destination_file}': {e}")
        logging.error(f"Error: an error occurred while copying '{source_file}' -> '{destination_file}': {e}")
    return False


# Publication on the JRC Research Data repository (https://research-data.ec.europa.eu, record
# "AI Public Health Surveillant"), through its InvenioRDM API. Set it to False to only copy the files
# to JRCOpenData_copy_location as before.
PUBLISH_TO_RESEARCH_DATA = True


def publish_to_research_data(basenames):
    """Publish the given output files of the pipeline as a new version of the Research Data record.

    A failure is reported and logged but does not stop the scheduler: the files are published again
    (as one version) at the next run.
    """
    from jrc_research_data import publish_new_version, ResearchDataError
    try:
        version_id = publish_new_version(
            [os.path.join(synk_location, basename) for basename in basenames],
            log=lambda message: (print(message), logging.info(message)))
        if version_id is None:
            logging.info("Research data: no new version published")
        return version_id
    except ResearchDataError as e:
        print(f"Error: publication on the research data repository failed: {e}")
        logging.error(f"Error: publication on the research data repository failed: {e}")
    except Exception as e:
        print(f"Error: unexpected error during the publication on the research data repository: {e}")
        logging.error(f"Error: unexpected error during the publication on the research data repository: {e}")
    return None


def job():

    try:

        # Log a message when the job starts
        print("\nJob started...")
        logging.info('\n')
        logging.info('Job started...')

        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        logging.info(now.strftime("%Y-%m-%d %H:%M:%S"))

        # The job is only run inside the service window. It normally fires at the beginning of it,
        # but this guard covers the cases where it does not: the script restarted at another hour
        # (see the os.execv at the bottom of this function), a job scheduled for a weekday that
        # fires on a weekend, or a machine that was asleep at the scheduled time.
        if not within_service_window(now):
            print("Job skipped: " + service_window_message(now) +
                  ". Nothing is run, the work is left to the next scheduled job.")
            logging.info("Job skipped: " + service_window_message(now))
            return

        # ... run your code ...
        print("Running llm_extraction_AutomatedDynamicPipeline.py from Py-RUN")
        logging.info("Running llm_extraction_AutomatedDynamicPipeline.py from Py-RUN")
        #print("CIAO")
        try:
            main(synk_location)
        except Exception as e:
            print(f"Error: '{e}' on main function call.")
            logging.error(f"Error: '{e}' on main function call.")

        # Log a message when the job finishes
        print('Job finished')
        logging.info('Job finished')

        # COPY FILES TO JRC-OpenData
        # NOTE: the _whoDons.csv copy has been removed: the harvesting is MedicalOfficial-only now,
        # so the DONs-only output is not produced anymore.

        # only the shareable ("_red") versions are published: the extractions without the full text,
        # the fileid and the media outlet of the articles, and the ontology without the
        # ekg:MediaReport nodes and the foundation:hasSignal links, i.e. without disclosing the
        # sources of the outbreaks. The two are joined on the record identifier of the outbreak
        # (the record_id column of the CSV, the rdfs:label of the event in the ontology).
        # Disabled since 2026-10-19: the files are now published on the JRC Research Data repository
        # (below). The copy to JRCOpenData_copy_location is kept here in case it is needed again.
        #for basename in ['OutputAnnotatedTexts-FT-LLM_red.csv', 'epidemicIE_red.ttl', 'epidemicIE_red.rdf']:
        #    copy_to_JRCOpenData(basename)

        # PUBLISH THE SAME FILES ON THE JRC RESEARCH DATA REPOSITORY, as a new version of the record
        # "AI Public Health Surveillant" (see jrc_research_data.py). No version is created when the
        # files are identical to the ones already published.
        if PUBLISH_TO_RESEARCH_DATA:
            publish_to_research_data(['OutputAnnotatedTexts-FT-LLM_red.csv', 'epidemicIE_red.ttl',
                                      'epidemicIE_red.rdf'])

        # schedule part:
        all_jobs = schedule.get_jobs()
        print(all_jobs)
        logging.info(all_jobs)
        logging.info('\n')

    except Exception as e:
        print(f"Caught an error: {e}")
        print("Bus error occurred, restarting in 30 seconds...")
        logging.error(f"Caught an error: {e}")
        logging.error("Bus error occurred, restarting in 30 seconds...")
        time.sleep(30)
        os.execv(sys.executable, ['python'] + sys.argv)

    #return schedule.CancelJob   # decomment this line and the next line if you want to run the job only once
    #schedule.clear()


now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
logging.info(now.strftime("%Y-%m-%d %H:%M:%S"))

# schedule part:
all_jobs = schedule.get_jobs()
print(all_jobs)
logging.info(all_jobs)

#schedule.every().minute.at(":17").do(job)  # this is for debuggin
#job()  # this is for debugging


def job_once():
    """The job, run only once: used for the catch-up run of a scheduler started in the morning."""
    job()
    return schedule.CancelJob


# The recurring schedule: the job starts at the opening of the service window, on working days only.
# "Europe/Amsterdam" is the same wall clock as the local time of this machine (Europe/Rome), which is
# the clock within_service_window() reads, so the two cannot disagree.
# It is registered in any case, whatever the moment the scheduler is started.
daily_start = SERVICE_WINDOW_START.strftime("%H:%M")
print("Job scheduled every working day at " + daily_start + " (Europe/Amsterdam), service window " +
      daily_start + "-" + SERVICE_WINDOW_END.strftime("%H:%M"))
logging.info("Job scheduled every working day at " + daily_start + " (Europe/Amsterdam)")

#schedule.every(10).minutes.do(job)
#schedule.every().hour.do(job)
#schedule.every().day.at("10:30").do(job)
#schedule.every().monday.do(job)
#schedule.every().wednesday.at("13:15").do(job)
#schedule.every().minute.at(":17").do(job)
#schedule.every().day.at("17:40", "Europe/Amsterdam").do(job)
#schedule.every().day.at("22:45", "Europe/Amsterdam").do(job)
#schedule.every().day.at("19:45", "Europe/Amsterdam").do(job)
schedule.every().monday.at(daily_start, "Europe/Amsterdam").do(job)
schedule.every().tuesday.at(daily_start, "Europe/Amsterdam").do(job)
schedule.every().wednesday.at(daily_start, "Europe/Amsterdam").do(job)
schedule.every().thursday.at(daily_start, "Europe/Amsterdam").do(job)
schedule.every().friday.at(daily_start, "Europe/Amsterdam").do(job)

# On top of that, when the scheduler is started inside the morning part of the service window (so
# that a whole run has the time to go through before the services close), a single catch-up run is
# fired 15 minutes later, instead of waiting for the next working day. It is a one-shot job: it
# cancels itself after the first execution, leaving only the schedule above.
morning_end = now.replace(hour=12, minute=0, second=0, microsecond=0)

if within_service_window(now) and now <= morning_end:
    new_time = now + datetime.timedelta(minutes=15)
    print("Started inside the morning window: a single catch-up run is fired at " +
          new_time.strftime("%H:%M"))
    logging.info("Started inside the morning window: single catch-up run at " +
                 new_time.strftime("%H:%M"))
    schedule.every().day.at(new_time.strftime("%H:%M"), "Europe/Amsterdam").do(job_once)
else:
    print("Not started inside the morning window (" + service_window_message(now) +
          "): waiting for the next working day at " + daily_start)
    logging.info("Not started inside the morning window (" + service_window_message(now) + ")")





all_jobs = schedule.get_jobs()
print(all_jobs)
logging.info(all_jobs)

while True:
    schedule.run_pending()
    #time.sleep(1)
    time.sleep(300)

