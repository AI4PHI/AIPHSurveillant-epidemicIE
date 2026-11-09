# Gradio demo: epidemiological information extraction with EpiQwen3.5-9B, shown as a knowledge graph.
# This is the code behind the public demo at https://jrc-ai-epidemicie.hf.space.
# (The previous Epi-Mistral-7B version of this app is in the git history.)
#
# Run:   python app.py                            (http://<host>:7860)
#        python app.py --share True               (public gradio.live link)
#        CUDA_VISIBLE_DEVICES=1 python app.py --port 7861
import os

# %%
# Optional runtime configuration (uncomment and adapt to your environment):
# The Hugging Face cache location can be set here if the default (~/.cache/huggingface) is not
# suitable, e.g. on a shared machine with limited home quota. Left to the HF default by design;
# set the HF_HOME / HF_HUB_CACHE environment variables before running if you need to relocate it.
# os.environ["HF_HUB_CACHE"] = "/path/to/huggingface/hub"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/path/to/huggingface/hub"
# os.environ["HF_HOME"] = "/path/to/huggingface/hub"

# %%
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# One GPU only: on this server direct GPU-to-GPU copies are silently corrupted, so a model split over
# 2 GPUs (device_map="auto") generates garbage. Override from the shell, e.g. CUDA_VISIBLE_DEVICES=1.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

##############################################

# %%
import argparse
import ast
import html
import json
import math
import re

import gradio as gr
import pandas as pd
import torch
from peft import PeftModel
from pyvis.network import Network
from transformers import AutoTokenizer

import serving_format

ie_model_id = "jattokatarratto/EpiQwen3.5-9B"   # LoRA adapter on Qwen/Qwen3.5-9B
#ie_model_id = "/path/to/EpiQwen3.5-9B"   # same release, local copy

title = "Information Extraction from Epidemiological Reports"

# Fields of one extracted outbreak record (order of the model output)
FIELDS = [
    "DiseaseLevel1", "DiseaseLevel2", "Country", "ISO", "OutbreakEpicenter",
    "CasesTotal", "CasesSuspected", "CasesProbable", "CasesConfirmed", "Deaths",
    "OutbreakStartYear", "OutbreakStartMonth", "OutbreakStartDay",
    "OutbreakDetectionYear", "OutbreakDetectionMonth", "OutbreakDetectionDay",
    "OutbreakVerificationYear", "OutbreakVerificationMonth", "OutbreakVerificationDay",
    "OutbreakEnd", "OutbreakEndYear", "OutbreakEndMonth", "OutbreakEndDay",
]
# Year/Month/Day triplets are shown as one date
DATES = {"OutbreakStart": "Start", "OutbreakDetection": "Detection",
         "OutbreakVerification": "Verification", "OutbreakEnd": "End"}
# Entity-like values get ONE node shared by all outbreaks of the report (e.g. the same disease in
# several countries); counts and dates stay attached to their own outbreak.
SHARED = {"DiseaseLevel1": "disease", "DiseaseLevel2": "disease", "Country": "place",
          "OutbreakEpicenter": "place"}
COLORS = {"outbreak": "#d62728", "disease": "#9467bd", "place": "#1f77b4",
          "count": "#ff7f0e", "date": "#2ca02c", "other": "#7f7f7f"}


# %%
def load_model(model_id):
    """
    EpiQwen3.5-9B is a LoRA adapter: load the base model it was trained on with the class used in
    training (Qwen3_5ForConditionalGeneration, NOT AutoModelForCausalLM) and merge the adapter.
    A merged / full model (no adapter_config.json) is loaded directly.
    """
    fmt = serving_format.find_serving_format(model_id)
    if fmt is None:
        raise SystemExit(f"No serving_prompt_format.json for {model_id}: this app needs a chat-format "
                         f"fine-tune (EpiQwen3.5-9B).")
    adapter_base = serving_format.adapter_base_model(model_id)
    weights = fmt["model_id"] if adapter_base is not None else model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    model = serving_format.load_model_class(weights).from_pretrained(
        weights, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    if adapter_base is not None:
        model = PeftModel.from_pretrained(model, model_id).merge_and_unload()
        torch.cuda.empty_cache()   # free the separate LoRA tensors after merging
    model.eval()
    print(f"Loaded {'adapter ' + model_id + ' on ' + weights if adapter_base else model_id} "
          f"on {model.device}")
    return model, tokenizer, fmt


def extract_records(text):
    """Run the model on one report; returns (list of record dicts, raw model answer)."""
    prompt = serving_format.build_chat_prompt(tokenizer, fmt, text)
    inputs = serving_format.tokenize_prompts(tokenizer, [prompt], device=model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             **serving_format.greedy_kwargs(tokenizer, fmt))
    answer = serving_format.decode_answer(tokenizer, out[0][inputs["input_ids"].shape[1]:], fmt)
    return parse_records(answer), answer


def parse_records(answer):
    """
    The model answers with a Python-style list of JSON strings, one per outbreak, NaN for missing
    values: ['{"DiseaseLevel1": "Cholera", ..., "Deaths": NaN}', '{...}']. If that fails (e.g. a
    truncated answer), fall back to every complete {...} object found in the text.
    """
    try:
        items = ast.literal_eval(answer.strip())
        return [json.loads(i) if isinstance(i, str) else dict(i) for i in items]
    except Exception:
        pass
    records = []
    for chunk in re.findall(r"\{[^{}]*\}", answer):
        try:
            records.append(json.loads(chunk))
        except Exception:
            continue
    return records


def fmt_value(v):
    """NaN / empty -> None; 3.0 -> '3'; anything else as a string."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    return s[:-2] if s.endswith(".0") and s[:-2].lstrip("-").isdigit() else s


def fmt_date(rec, prefix):
    """Year/Month/Day fields -> 'YYYY-MM-DD' (partial dates allowed), or None."""
    y, m, d = (fmt_value(rec.get(prefix + p)) for p in ("Year", "Month", "Day"))
    if y is None:
        return None
    parts = [y]
    if m is not None and m.isdigit():
        parts.append(m.zfill(2))
        if d is not None and d.isdigit():
            parts.append(d.zfill(2))
    return "-".join(parts)


def outbreak_label(i, rec):
    disease = fmt_value(rec.get("DiseaseLevel1")) or "Outbreak"
    country = fmt_value(rec.get("Country"))
    return f"#{i} {disease}" + (f" - {country}" if country else "")


def records_to_graph(records):
    net = Network(directed=True, height="620px", width="100%", cdn_resources="remote")

    def add_value(node_id, label, kind):
        net.add_node(node_id, label=label, title=label, shape="ellipse", color=COLORS[kind])

    for i, rec in enumerate(records, 1):
        ob = f"outbreak-{i}"
        net.add_node(ob, label=outbreak_label(i, rec), shape="box", color=COLORS["outbreak"],
                     font={"color": "white"})

        for field in FIELDS:
            if any(field.startswith(p) and field != p for p in DATES):   # Year/Month/Day
                continue
            v = fmt_value(rec.get(field))
            if v is None:
                continue
            if field in SHARED:
                node, kind = f"{SHARED[field]}:{v.lower()}", SHARED[field]
            elif field == "ISO":
                node, kind = f"place:iso:{v.upper()}", "place"
            elif field.startswith("Cases") or field == "Deaths":
                node, kind = f"{ob}:{field}", "count"
            else:
                node, kind = f"{ob}:{field}", "other"
            add_value(node, v, kind)
            net.add_edge(ob, node, label=field, title=field)

        for prefix, name in DATES.items():
            d = fmt_date(rec, prefix)
            if d is not None:
                node = f"{ob}:{prefix}Date"
                add_value(node, d, "date")
                net.add_edge(ob, node, label=f"{name} date", title=f"{prefix} date")

    if not records:
        net.add_node("none", label="No outbreak information found", shape="box",
                     color=COLORS["other"])

    net.repulsion(node_distance=180, central_gravity=0.15, spring_length=160,
                  spring_strength=0.05, damping=0.09)
    net.set_edge_smooth("dynamic")

    # the pyvis page goes in an iframe srcdoc: escape it as an HTML attribute value
    return (f'<iframe style="width:100%; height:640px; border:0;" sandbox="allow-scripts allow-same-origin" '
            f'srcdoc="{html.escape(net.generate_html(), quote=True)}"></iframe>')


def records_to_table(records):
    rows = []
    for i, rec in enumerate(records, 1):
        row = {"#": i}
        for field in ("DiseaseLevel1", "DiseaseLevel2", "Country", "ISO", "OutbreakEpicenter",
                      "CasesTotal", "CasesSuspected", "CasesProbable", "CasesConfirmed", "Deaths"):
            row[field] = fmt_value(rec.get(field)) or ""
        for prefix, name in DATES.items():
            row[f"{name} date"] = fmt_date(rec, prefix) or ""
        row["OutbreakEnd"] = fmt_value(rec.get("OutbreakEnd")) or ""
        rows.append(row)
    return pd.DataFrame(rows)


def text_to_graph(text):
    if not text or not text.strip():
        raise gr.Error("Enter the text of an outbreak report first.")
    records, answer = extract_records(text)
    if not records and answer.strip() not in ("[]", ""):
        gr.Warning("The model answer could not be parsed - see the raw output below.")
    # The model was trained on outbreak reports only: on unrelated text it may still return a record
    # with just a place. A record without any disease is not an outbreak.
    n_all = len(records)
    records = [r for r in records if fmt_value(r.get("DiseaseLevel1")) or fmt_value(r.get("DiseaseLevel2"))]
    n = len(records)
    status = (f"**{n} outbreak{'s' if n != 1 else ''} extracted.**" if n
              else "**No outbreak information found in the text.**")
    if n_all > n:
        status += f" ({n_all - n} record(s) without a disease ignored - see the raw output.)"
    # NaN is not valid JSON for the browser: show missing values as null
    clean = [{k: (None if fmt_value(v) is None else v) for k, v in r.items()} for r in records]
    return status, records_to_graph(records), records_to_table(records), clean, answer


#The Disease Outbreaks News database is referenced in the following publication:
#Carlson CJ, Boyce MR, Dunne M, Graeden E, Lin J, et al. (2023) The World Health Organization’s Disease Outbreak News: A retrospective database. PLOS Global Public Health 3(1): e0001083. https://doi.org/10.1371/journal.pgph.0001083.
#The database is available at: https://journals.plos.org/globalpublichealth/article?id=10.1371/journal.pgph.0001083

description = """
This app uses [EpiQwen3.5-9B](https://huggingface.co/jattokatarratto/EpiQwen3.5-9B),
a Large Language Model (LLM) obtained by fine-tuning (LoRA) the pre-trained
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) model for the task of epidemiological information extraction.

The model has been fine-tuned in the [JRC Big Data Analytics Platform](https://jeodpp.jrc.ec.europa.eu/bdap/)
on the [Disease Outbreaks News database](https://doi.org/10.1371/journal.pgph.0001083). It reads an outbreak report and
extracts **one structured record per outbreak mentioned** (a report can describe several diseases or countries): disease, country, epicenter,
case numbers (total, suspected, probable, confirmed), deaths, and the start, detection, verification and end dates of the outbreak.

Enter a piece of text (an epidemiological report) in the text box and click **Extract**.
The extracted outbreaks are shown as a graph (outbreaks in red; shared diseases and places link outbreaks of the same report),
as a table and as JSON. You can also use the pre-loaded examples to see how the app works.
"""

references = """
## References

1. Carlson CJ, Boyce MR, Dunne M, Graeden E, Lin J, Abdellatif YO, et al. (2023) The World Health Organization’s Disease Outbreak News: A retrospective database. PLOS Global Public Health, 3(1): e0001083. [doi:10.1371/journal.pgph.0001083](https://doi.org/10.1371/journal.pgph.0001083).
2. Consoli S, Bertolini L, Biazzo I, Spagnolo L, Stilianakis NI (2027) From Outbreak Reports to a Daily-Updated Epidemiological Knowledge Graph: Fine-Tuned Large Language Models for Structured Information Extraction in the Post-COVID-19 Era. Frontiers in Public Health (under revision).
3. Consoli S, Bertolini L, Stefanovitch N, Spagnolo L, Espinosa L, Stilianakis NI (2026) Generative AI for Structured Epidemiological Information Extraction: Comparing In-Context Learning and Fine-Tuning Approaches. PLOS Digital Health (under revision).
4. Consoli S, Markov P, Stilianakis NI, Bertolini L, Puertas-Gallardo A, Ceresa M (2024) Epidemic Information Extraction for Event-Based Surveillance Using Large Language Models. Lecture Notes in Networks and Systems, 1011, Springer. [doi:10.1007/978-981-97-4581-4_17](https://doi.org/10.1007/978-981-97-4581-4_17).
5. Soille P, Burger A, De Marchi D, Kempeneers P, Rodriguez D, Syrris V, et al (2018) A versatile data-intensive computing platform for information retrieval from big geospatial data. Future Generation Computer Systems, 81, 30–40. [doi:10.1016/j.future.2017.11.007](https://doi.org/10.1016/j.future.2017.11.007).
"""

examples = [
    # one outbreak
    ["A 48-year-old unvaccinated man travelling in Bolivar state (Venezuela) became ill on 23 September 1999 and returned to California on 25 September. He was hospitalized on 27 September with fulminant hepatitis and renal failure, and died on 4 October. Yellow fever was confirmed by immunohistochemistry and PCR. The Venezuelan authorities were informed and a field investigation is under way. This event serves once more to illustrate the vital importance for travellers to have yellow fever vaccination before visiting endemic areas."],
    ["A man travelling in the Bolivar state of Venezuela was hospitalized on 27 September 1999 with fulminant hepatitis and renal failure and died on 4 October of Yellow fever, as confirmed by immunohistochemistry and PCR. The Venezuelan authorities were informed and a field investigation is under way. This event serves once more to illustrate the vital importance for travellers to have yellow fever vaccination before visiting endemic areas."],
    ["As of end of February 1999, 1 confirmed case of sylvatic yellow fever was reported to the Pan American Health Organization (PAHO/WHO). This case is the first national reported case of yellow fever since November 1998. The case occurred in the Native Indian village of Aldeia Cuoto Magalhaes, in the county of Campinapolis, Mato Grosso State. The patient who recovered was a 5 year old girl without a history of yellow fever vaccination. Source: Ministry of Health, Brazil."],
    ["A suspected case of Ebola virus was reported on 6 December 1996  to WHO. The patient was a 25-year-old male from a village in neighbouring Liberia who was admitted to hospital in the Tabou Prefecture (south of Tai National Park). The man appears to be recovering from the illness. The diagnosis of Ebola virus infection was confirmed by serological tests carried out at the Institute Pasteur in Paris. A WHO team including an expert from the Institut Pasteur has investigated the case in close cooperation with the Ministry of Health, Médecins Sans Frontières, Epicentre (Paris) and UNHCR. The team has ensured the patient is properly isolated in the hospital and that all precautions have been taken to prevent spread of the infection. The team has also investigated hospital and village contacts of the confirmed case and visited the household of the case which is located in Plibo Village Maryland County, Liberia. The man lived there with three other men and two women. The team found two of the three men were sick with fever and extreme fatigue which could be signs of Ebola virus infection. Both have been isolated in their home. The third man is well. Of the two women one has been hospitalized in Côte d'Ivoire with symptoms which could be Ebola fever. The other one is well. Samples of blood from the three suspect cases are under investigation. WHO is presently organizing an international study of the epidemiology and ecology of the virus in the Tai Forest Region of Côte d'Ivoire. The reservoir and vector of Ebola virus in nature remains unknown. WHO does not recommend that any special measures be instituted for international travel (in relation to the Ebola incident) to and from Côte d'Ivoire and Liberia."],
    # several outbreaks in one report
    ["11 October 2006. As of 11 October, 4 cases of botulism associated with commercial carrot juice have been reported in the states of Georgia and Florida in the United States. Two cases have been reported in Toronto, Canada. All cases had consumed the same brand of carrot juice and no fatalities have been reported to date. The United States Centers for Disease Prevention and Control and the Canadian Food Inspection Agency have issued advisory information. The International Food Safety Authorities Network (INFOSAN) has notified Mexico and Hong Kong SAR where the carrot juice has also been exported."],
    ["27 March 1997 Disease Outbreak Reported As at 23 March 1997, a total of 24 798 cases with 2 933 deaths from epidemic meningitis had been reported in Benin, Burkina Faso, Gambia, Ghana, Mali, Niger, and Togo. The epidemic in Burkina Faso accounted for 42% (10,429 cases with 1 318 deaths) and in Ghana for 35% where the cumulative number of cases more than doubled from 3 757 (411 deaths) on 13 March to 8 671 (902 deaths) on 23 March. Most (95%) of the cases in Ghana have been reported in Upper East Region (6 364 cases, 497 deaths) and Northern Region (1 867 cases, 281 deaths). Cases have also been detected in Ashanti, Brong Ahafo and Upper West Regions."],
]


def build_demo():
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)   # not gr.HTML: it lays out each link as a separate flex item (extra spaces)
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(label="Text", placeholder="Enter the text of an outbreak report here.",
                                  lines=12)
                with gr.Row():
                    clear = gr.ClearButton(value="Clear")
                    submit = gr.Button("Extract", variant="primary")
                status = gr.Markdown()
            with gr.Column(scale=3):
                graph = gr.HTML(label="Extracted graph")
        table = gr.Dataframe(label="Extracted outbreaks", wrap=True)
        with gr.Accordion("JSON output", open=False):
            records_json = gr.JSON(label="Extracted records")
            raw = gr.Textbox(label="Raw model output", lines=8)

        outputs = [status, graph, table, records_json, raw]
        gr.Examples(examples=examples, inputs=[text], outputs=outputs, fn=text_to_graph,
                    cache_examples=strtobool(args.cache_examples), cache_mode="lazy")
        submit.click(text_to_graph, inputs=[text], outputs=outputs)
        clear.add([text, status, graph, table, records_json, raw])
        gr.Markdown(references)
    return demo


def strtobool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio demo of EpiQwen3.5-9B")
    parser.add_argument("--model_id", type=str, default=ie_model_id,
                        help="HF repo or local dir: LoRA adapter (loaded on its base model) or merged model")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="default: recommended value in serving_prompt_format.json (2048)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", type=str, default="False", help="public gradio.live link")
    parser.add_argument("--cache_examples", type=str, default="True",
                        help="cache the example outputs (computed on first click, then instant)")
    args = parser.parse_args()

    model, tokenizer, fmt = load_model(args.model_id)
    if args.max_new_tokens is None:
        args.max_new_tokens = fmt.get("recommended_generation", {}).get("max_new_tokens", 2048)

    # one request at a time: a single model on a single GPU
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=strtobool(args.share))
