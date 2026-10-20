"""Helpers to query LLM chat-completion APIs over a list of texts.

Supports OpenAI and OpenAI-compatible gateways (configurable base URLs). Features:
  - built-in retry with back-off on rate-limit and transient errors;
  - optional on-disk caching of results, keyed by model/temperature/prompt, so a crash does not
    lose completed calls;
  - a uniform ``call_model`` / ``process_list`` interface that takes the concrete API handler as a
    delegate.

In the current pipeline the extraction itself is done by the local fine-tuned model
(see ``llm_extraction_AutomatedDynamicPipeline``); this module is kept for the remote-API backends
and auxiliary experiments. API tokens are read from local ``*.key`` files or environment variables
and are never hardcoded. The base URLs of the institution-internal gateways default to empty and
must be provided via the GPTJRC_API_BASE / GPTEC_API_BASE environment variables.
"""

import os
import sys

import openai
import json
import time
from tqdm import tqdm

import logging

from functools import partial
import pandas as pd

import tiktoken




def encoding_getter(encoding_type: str):
    """
    Returns the appropriate encoding based on the given encoding type (either an encoding string or a model name).

    tiktoken supports three encodings used by OpenAI models:

    Encoding name	OpenAI models
    cl100k_base	gpt-4, gpt-3.5-turbo, text-embedding-ada-002
    p50k_base	Codex models, text-davinci-002, text-davinci-003
    r50k_base (or gpt2)	GPT-3 models like davinci

    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    """
    if "k_base" in encoding_type:
        return tiktoken.get_encoding(encoding_type)
    else:
        try:
            my_enc = tiktoken.encoding_for_model(encoding_type)
            return my_enc
        except Exception as err:
            my_enc = tiktoken.get_encoding("cl100k_base")   #default for gpt-4, gpt-3.5-turbo
            return my_enc

def tokenizer(string: str, encoding_type: str) -> list:
    """
    Returns the tokens in a text string using the specified encoding.
    """
    encoding = encoding_getter(encoding_type)
    tokens = encoding.encode(string)
    return tokens

def token_counter(string: str, encoding_type: str) -> int:
    """
    Returns the number of tokens in a text string using the specified encoding.
    """
    num_tokens = len(tokenizer(string, encoding_type))
    return num_tokens


# ### OPENAI API

def setup_openai(org=None, key=None):
    if org is not None:
        openai.organization = org
    # else:
    #     openai.organization = os.getenv("OPENAI_API_ORGANIZATION")

    if key is not None:
        openai.api_key = key
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    #
    print(model_list_openai())

def api_call_openai(prompt: str, input_text: str, model: str, temperature: int, timeout_retry: int=5, delimiter: str = "```", InContextExamples: list[[str]] = [], debug=False):
    """ call openai API, with a retry in case of RateLimitError """

    if not(prompt) or prompt.strip=="" or not(input_text) or input_text.strip=="":
        logging.warning("No text or promt supplied! Skypping it!")
        return None

    if delimiter and len(delimiter)>0:
        input_text = delimiter + input_text + delimiter

    response = None

    myMessages = []
    if InContextExamples:
        for row in InContextExamples:
            myMessages.append({"role": "system", "content": prompt})
            for indCol, colVal in enumerate(row):
                if indCol == 0:
                    if delimiter and len(delimiter) > 0:
                        myMessages.append({"role": "user", "content": (delimiter + colVal + delimiter)})
                    else:
                        myMessages.append({"role": "user", "content": colVal})
                elif indCol == 1:
                    myMessages.append({"role": "assistant", "content": colVal})

    myMessages.append({"role": "system", "content": prompt})
    myMessages.append({'role': 'user', 'content': input_text})

    max_retries = 50
    iteration = 1
    while response is None and max_retries > 0:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                # messages=[
                #     {"role": "system", "content": prompt},
                #     {'role': 'user', 'content': input_text},
                # ],
                messages=myMessages,
                temperature=temperature,
                #max_tokens=32000,  #it gives error
                #max_response_tokens=32000 #it gives error
            )
        except openai.error.RateLimitError as e:
            response = None
            max_retries = max_retries - 1
            print(e)
            nt = token_counter((prompt + input_text), model)
            print("Model "+str(model)+" - Length of overall prompt message ", str(nt))
            print("current iteration ", iteration)
            print("try other ", max_retries, " times")
            print("sleeping", int(iteration * timeout_retry), "s")
            print(time.sleep(int(iteration * timeout_retry)))
            iteration = iteration + 1
        except Exception as err:
            response = None
            max_retries = max_retries - 1
            print(err)
            nt = token_counter((prompt + input_text), model)
            print("Model " + str(model) + " - Length of overall prompt message ", str(nt))
            print("current iteration ", iteration)
            print("try other ", max_retries, " times")
            print("sleeping", int(iteration*timeout_retry), "s")
            print(time.sleep(int(iteration*timeout_retry)))
            iteration = iteration + 1

    if (response == None) and (max_retries <= 0):
        print("\n")
        print(prompt + input_text)
        print("\n")
        print("\nTried many times and did not succeed, there is something strange. Check the problem...exiting now\n")
        sys.exit()

    return response

def model_list_openai():
    return openai.Model.list()


### GPT@EC API
# Base URL of the GPT@EC gateway. This is an institution-internal LLM gateway; replace it with your
# own OpenAI-compatible API base by setting the GPTEC_API_BASE environment variable.
GPTEC_API_BASE = os.environ.get("GPTEC_API_BASE", "")


def setup_gptec(token=None):
    if token is None:
        token = os.environ.get('key_gptec', '')
    openai.organization = ""
    openai.api_key = token
    openai.api_base = GPTEC_API_BASE
    print(model_list_gptec())


def api_call_gptec(prompt: str, input_text: str, model: str, temperature: int, timeout_retry: int = 5, delimiter: str = "```", InContextExamples: list[[str]] = [], debug=False):
    if not (prompt) or prompt.strip == "" or not (input_text) or input_text.strip == "":
        logging.warning("No text or prompt supplied! Skipping it!")
        return None

    if delimiter and len(delimiter) > 0:
        input_text = delimiter + input_text + delimiter

    response = None
    myMessages = []
    if InContextExamples:
        for row in InContextExamples:
            myMessages.append({"role": "system", "content": prompt})
            for indCol, colVal in enumerate(row):
                if indCol == 0:
                    if delimiter and len(delimiter) > 0:
                        myMessages.append({"role": "user", "content": (delimiter + colVal + delimiter)})
                    else:
                        myMessages.append({"role": "user", "content": colVal})
                elif indCol == 1:
                    myMessages.append({"role": "assistant", "content": colVal})

    myMessages.append({"role": "system", "content": prompt})
    myMessages.append({'role': 'user', 'content': input_text})

    max_retries = 50
    iteration = 1
    while response is None and max_retries > 0:
        try:
            response = openai.ChatCompletion.create(
                headers={"Authorization": "Bearer " + openai.api_key},
                model=model,
                messages=myMessages,
                temperature=temperature,
            )
        except openai.error.RateLimitError as e:
            response = None
            max_retries = max_retries - 1
            print(e)
            nt = token_counter((prompt + input_text), model)
            print("Model " + str(model) + " - Length of overall prompt message ", str(nt))
            print("current iteration ", iteration)
            print("try other ", max_retries, " times")
            print("sleeping", int(iteration * timeout_retry), "s")
            print(time.sleep(int(iteration * timeout_retry)))
            iteration = iteration + 1
            print("\nprompt:")
            print(prompt)
            print("\ninput_text:")
            print(input_text)
            if max_retries in [45, 40, 35, 30, 25, 20, 15, 10, 5]:
                trim_target = int(len(input_text) * 0.85)
                trimmed = input_text[:trim_target]
                last_dot = trimmed.rfind('.')
                if last_dot > len(trimmed) // 2:
                    trimmed = trimmed[:last_dot + 1]
                input_text = trimmed + delimiter
                print(f"Trimmed input_text to {len(input_text)} chars")
                myMessages = []
                myMessages.append({"role": "system", "content": prompt})
                myMessages.append({'role': 'user', 'content': input_text})
        except Exception as err:
            response = None
            max_retries = max_retries - 1
            print(err)
            nt = token_counter((prompt + input_text), model)
            print("Model " + str(model) + " - Length of overall prompt message ", str(nt))
            print("current iteration ", iteration)
            print("try other ", max_retries, " times")
            print("sleeping", int(iteration * timeout_retry), "s")
            print(time.sleep(int(iteration * timeout_retry)))
            iteration = iteration + 1
            print("\nprompt:")
            print(prompt)
            print("\ninput_text:")
            print(input_text)
            if max_retries in [45, 40, 35, 30, 25, 20, 15, 10, 5]:
                trim_target = int(len(input_text) * 0.85)
                trimmed = input_text[:trim_target]
                last_dot = trimmed.rfind('.')
                if last_dot > len(trimmed) // 2:
                    trimmed = trimmed[:last_dot + 1]
                input_text = trimmed + delimiter
                print(f"Trimmed input_text to {len(input_text)} chars")
                myMessages = []
                myMessages.append({"role": "system", "content": prompt})
                myMessages.append({'role': 'user', 'content': input_text})

    if (response == None) and (max_retries <= 0):
        print("\n")
        print(prompt + input_text)
        print("\n")
        print("\nTried many times and did not succeed, there is something strange. Check the problem...exiting now\n")
        sys.exit()

    return response


def model_list_gptec():
    return openai.Model.list()


### GPT@JRC API
# Base URL of the GPT@JRC gateway. This is an institution-internal LLM gateway; replace it with your
# own OpenAI-compatible API base by setting the GPTJRC_API_BASE environment variable.
GPTJRC_API_BASE = os.environ.get("GPTJRC_API_BASE", "")


def setup_gptjrc(token=None):
    if token is None:
        token=os.getenv("GPTJRC_TOKEN")
    openai.organization = ""
    openai.api_key = token
    #openai.api_type = "open_ai"
    openai.api_base = GPTJRC_API_BASE
    #
    print(model_list_gptjrc())



def api_call_gptjrc(prompt: str, input_text: str, model: str, temperature: int, timeout_retry: int=5, delimiter: str = "```", InContextExamples: list[[str]] = [], debug=False, request_timeout: int=1200):


    if not (prompt) or prompt.strip=="" or not(input_text) or input_text.strip=="":
        logging.warning("No text or promt supplied! Skypping it!")
        return None

    if delimiter and len(delimiter)>0:
        input_text = delimiter + input_text + delimiter

    response = None

    myMessages = []
    if InContextExamples:
        for row in InContextExamples:
            myMessages.append({"role": "system", "content": prompt})
            for indCol, colVal in enumerate(row):
                if indCol == 0:
                    if delimiter and len(delimiter) > 0:
                        myMessages.append({"role": "user", "content": (delimiter + colVal + delimiter)})
                    else:
                        myMessages.append({"role": "user", "content": colVal})
                elif indCol == 1:
                    myMessages.append({"role": "assistant", "content": colVal})

    myMessages.append({"role": "system", "content": prompt})
    myMessages.append({'role': 'user', 'content': input_text})

    max_retries = 50
    iteration = 1
    while response is None and max_retries>0:
        try:
            # if InContextExamples:
            #     response = openai.ChatCompletion.create(
            #         headers={"Authorization": "Bearer " + openai.api_key},
            #         model=model,
            #         messages=[
            #             {"role": "system", "content": prompt},
            #             {'role': 'user', 'content': InContextExamples[0][0]},
            #             {'role': 'assistant', 'content': InContextExamples[0][1]},
            #             {"role": "system", "content": prompt},
            #             {'role': 'user', 'content': InContextExamples[1][0]},
            #             {'role': 'assistant', 'content': InContextExamples[1][1]},
            #             {"role": "system", "content": prompt},
            #             {'role': 'user', 'content': InContextExamples[2][0]},
            #             {'role': 'assistant', 'content': InContextExamples[2][1]},
            #             {"role": "system", "content": prompt},
            #             {'role': 'user', 'content': input_text},
            #         ],
            #         temperature=temperature,
            #         # max_tokens=4000, #20000, #32000,  #it gives error
            #         # max_response_tokens=32000 #it gives error
            #     )
            # else:
            response = openai.ChatCompletion.create(
                headers={"Authorization": "Bearer "+openai.api_key},
                model=model,
                messages=myMessages,
                temperature=temperature,
                request_timeout=request_timeout,
            )

        except openai.error.RateLimitError as e:
            response = None
            max_retries = max_retries - 1
            print(e)
            nt = token_counter((prompt + input_text), model)
            print("Model " + str(model) + " - Length of overall prompt message ", str(nt))
            print("current iteration ", iteration)
            print("try other ", max_retries, " times")
            print("sleeping", int(iteration*timeout_retry), "s")
            print(time.sleep(int(iteration*timeout_retry)))
            iteration = iteration + 1
            print("\npromt:")
            print(prompt)
            print("\ninput_text:")
            print(input_text)
        except Exception as err:
            response = None
            max_retries = max_retries - 1
            print(err)
            nt = token_counter((prompt + input_text), model)
            print("Model " + str(model) + " - Length of overall prompt message ", str(nt))
            print("current iteration ", iteration)
            print("try other ", max_retries, " times")
            print("sleeping", int(iteration * timeout_retry), "s")
            print(time.sleep(int(iteration * timeout_retry)))
            iteration = iteration + 1
            print("\npromt:")
            print(prompt)
            print("\ninput_text:")
            print(input_text)
            if max_retries in [45, 40, 35, 30, 25, 20, 15, 10, 5]:
                trim_target = int(len(input_text) * 0.85)
                trimmed = input_text[:trim_target]
                last_dot = trimmed.rfind('.')
                if last_dot > len(trimmed) // 2:
                    trimmed = trimmed[:last_dot + 1]
                input_text = trimmed + delimiter
                print(f"Trimmed input_text to {len(input_text)} chars")
                myMessages = []
                myMessages.append({"role": "system", "content": prompt})
                myMessages.append({'role': 'user', 'content': input_text})


    if (response == None) and (max_retries <= 0):
        print("\n")
        print(prompt + input_text)
        print("\n")
        print("\nTried many times and did not succeed, there is something strange. Check the problem...exiting now\n")
        sys.exit()

    return response

def model_list_gptjrc():
    return openai.Model.list()


### DGLC API

def clean_gpt_out(output_text :str):

    if "From the text below, delimited by triple quotes, extract the following items: 1 - The name of the virus that has caused the outbreak" in output_text:
        print("debug")

    if "<|assistant|>" in output_text:
        output_text = output_text.split("<|assistant|>")[0].strip()
    if "<|prompt|>" in output_text:
        output_text = output_text.split("<|prompt|>")[0].strip()
    if "<|prompter|>" in output_text:
        output_text = output_text.split("<|prompter|>")[0].strip()
    if "<|answer|>" in output_text:
        output_text = output_text.split("<|answer|>")[0].strip()
    if "<|im_end|>" in output_text:
        output_text = output_text.split("<|im_end|>")[0].strip()
    if "<|endofextract|>" in output_text:
        output_text = output_text.split("<|endofextract|>")[0].strip()
    if "<br>" in output_text:
        output_text = output_text.split("<br>")[0].strip()

    if "<|/assistant|>" in output_text:
        output_text = output_text.split("<|/assistant|>")[0].strip()
    if "<|/prompt|>" in output_text:
        output_text = output_text.split("<|/prompt|>")[0].strip()
    if "<|/prompter|>" in output_text:
        output_text = output_text.split("<|/prompter|>")[0].strip()
    if "<|/answer|>" in output_text:
        output_text = output_text.split("<|/answer|>")[0].strip()
    if "<|/im_end|>" in output_text:
        output_text = output_text.split("<|/im_end|>")[0].strip()
    if "<|/endofextract|>" in output_text:
        output_text = output_text.split("<|/endofextract|>")[0].strip()
    if "</br>" in output_text:
        output_text = output_text.split("</br>")[0].strip()

    if "</|assistant|>" in output_text:
        output_text = output_text.split("</|assistant|>")[0].strip()
    if "</|prompt|>" in output_text:
        output_text = output_text.split("</|prompt|>")[0].strip()
    if "</|prompter|>" in output_text:
        output_text = output_text.split("</|prompter|>")[0].strip()
    if "</|answer|>" in output_text:
        output_text = output_text.split("</|answer|>")[0].strip()
    if "</|im_end|>" in output_text:
        output_text = output_text.split("</|im_end|>")[0].strip()
    if "</|endofextract|>" in output_text:
        output_text = output_text.split("</|endofextract|>")[0].strip()

    while "```" in output_text:
        output_text = output_text.replace("```", " ")

    while "  " in output_text:
        output_text = output_text.replace("  ", " ")

    return output_text


# def setup_dglc(key=None):
#     if key is None:
#         ACCESS_KEY = os.getenv("DGL_TOKEN")
#     else:
#         ACCESS_KEY=key
#
#     client_dglc.login(ACCESS_KEY)
#
#     #list available models
#     models_available = model_list_dglc()
#     print("DGLC - available models = "+str(models_available))
#
#     # chat_id = client_dglc.create_chat()
#     # # Or continue the previous one
#     # # chat_id = client_dglc.continue_chat(args.chat_id)
#     # print("\nCHAT_ID dglc", chat_id)


# def api_call_dglc(prompt: str, input_text: str, model: str, temperature: float, timeout_retry: int =5, delimiter: str = "```", InContextExamples: list[[str]] = [], debug=False):
#
#     # if model == "gpt-3.5-turbo":
#     #     model = "OA_GPT3.5"
#
#
#     if not(input_text) or input_text.strip=="" or not(prompt) or prompt.strip=="":
#         logging.warning("No text or promt supplied! Skypping it!")
#         return None
#
#     message = ""
#     if InContextExamples:
#         for row in InContextExamples:
#             message = message + prompt
#             for indCol, colVal in enumerate(row):
#                 if indCol == 0:
#                     if delimiter and len(delimiter) > 0:
#                         message = message + delimiter + colVal + delimiter
#                     else:
#                         message = message + colVal
#                 elif indCol == 1:
#                     message = message + " \n" + colVal + " \n"
#
#     if delimiter and len(delimiter) > 0:
#         message = prompt + delimiter + input_text + delimiter
#     else:
#         message = prompt + "\n" + input_text
#
#     if debug:
#         print("\n")
#         print(message)
#
#     chat_id = client_dglc.create_chat()
#     # Or continue the previous one
#     # chat_id = client.continue_chat(args.chat_id)
#     if debug:
#         print("\nCHAT_ID dglc", chat_id)
#
#     try:
#         events = client_dglc.send_message(message, model, temp=temperature)
#     except Exception as err:
#         logging.error(f'FAILED api_call_dglc WITH MESSAGE: \'{message}\' \nMODEL: {model}; \n\tError: {err}')
#
#     events= None
#     max_retries = 50
#     iteration = 1
#     while events is None and max_retries > 0:
#         try:
#             events = client_dglc.send_message(message, model, temp=temperature)
#         except Exception as err:
#             events = None
#             max_retries = max_retries - 1
#             print(err)
#             nt = token_counter((message), model)
#             print("Model " + str(model) + " - Length of overall prompt message ", str(nt))
#             print("current iteration ", iteration)
#             print("try other ", max_retries, " times")
#             print("sleeping", int(iteration * timeout_retry), "s")
#             print(time.sleep(int(iteration * timeout_retry)))
#             iteration = iteration + 1
#
#     if (events == None) and (max_retries <= 0):
#         print("\n")
#         print(message)
#         print("\n")
#         print("\nTried many times and did not succeed, there is something strange. Check the problem...exiting now\n")
#         sys.exit()
#
#     if events:
#         event = [str(x) for x in events]
#         # The message is streamed token by token
#         # for event in events:
#         #    print(event, end="", flush=True)
#         if event:
#             event = event[-1]
#         else:
#             event = None
#
#         if debug:
#             print("\nAPI CALL ANSWER:")
#             print(event)
#             print("\n")
#
#     else:
#         event = None
#
#     return event

# def model_list_dglc():
#     return client_dglc.get_available_models()


### CALLING MODELS


def call_model_with_caching(input_text: str, prompt: str, model: str, temperature: int, handler,
                            map_query_input_output: dict, cache_fp: str, timeout_retry: int =5, delimiter: str = "```", InContextExamples: list[[str]] = [], verbose: bool = True):
    """ call openai's API but take care of caching of results
    input_text: input text
    prompt: prompt
    model: model name (as parameter of the query)
    temperature: temperature (0: precise, 1: creative)
    handler: delegate function that will make the call (not necessarily only OpenAI, could be any one)
    map_query_input_output: cache dict containing already processed data
    cache_fp: file to which write content of cache after each call
    """

    if not(input_text) or input_text.strip=="" or not(prompt) or prompt.strip=="":
        logging.warning("No text or promt supplied! Skypping it!")
        return None

    # try to read cache

    if map_query_input_output is not None:
        key = model + "__" + str(temperature) + "__" + prompt

        if key in map_query_input_output:
            if input_text in map_query_input_output[key]:
                output = map_query_input_output[key][input_text]
                # if input_text.strip() == "":
                #     print("here")

                # if handler == api_call_dglc:
                #     output = clean_gpt_out(output) #clean output

                if verbose:
                    print("RETRIEVED CACHED RESULT FOR:\n", prompt, "\n", delimiter, input_text, delimiter, "\n=>\n", output, "\n")

                return output

    #  call

    response = None

    try:
        response = handler(prompt, input_text, model, temperature, timeout_retry, delimiter, InContextExamples)
    except Exception as err:
        logging.error(f'FAILED WITH PROMPT: \'{prompt}\' \nLEN_TEXT: {len(input_text)}, \nTEXT: {(input_text)}, \nMODEL: {model}; \n\tError: {err}')
    #else:
    #    # logging.warning(f'INDEX: \'{SOURCE_INDEX}\' Inserted {inserted} rows out of {num_lines} rows [{round((inserted/num_lines)*100, 2)}%]')
    #    break

    if response:
        if isinstance(response, str):
            output_text = response
        else:
            output_text = response['choices'][0]['message']['content']


        # if handler == api_call_dglc:
        #     output_text = clean_gpt_out(output_text)  # clean output


        # write to cache

        if map_query_input_output is not None:
            if not key in map_query_input_output:
                map_query_input_output[key] = {}

            if output_text:
                if output_text != "":
                    map_query_input_output[key][input_text] = output_text

                    with open(cache_fp, "w") as f:
                        json.dump(map_query_input_output, f)

        if verbose:
            print("API CALL REPLY FOR:\n", prompt, "\n", delimiter, input_text, delimiter, "\n=>\n", output_text, "\n")

        return output_text

    else:
        return None


def call_model(input_text: str, prompt: str, model: str, temperature: int, handler, timeout_retry: int =5, delimiter: str = "```", InContextExamples: list[[str]] = [],
               verbose: bool = True):
    """ call openai's API but take care of caching of resuts
    input_text: input text
    prompt: prompt
    model: model name (as parameter of the query)
    temperature: temperature (0: precise, 1: creative)
    handler: delegate function that will make the call (not necessarily only OpenAI, could be any one)
    """


    if not(input_text) or input_text.strip=="" or not(prompt) or prompt.strip=="":
        logging.warning("No text or promt supplied! Skypping it!")
        return None

    return call_model_with_caching(input_text, prompt, model, temperature, handler, None, None, timeout_retry, delimiter, InContextExamples, verbose)



def process_list(list_input_text: list[str], prompt: str, service_provider: str, model: str, temperature: int,
                 cache_prefix_fp: str = None, delimiter: str = "```", InContextExamples: list[[str]] = []):
    """ process a list of text with a prompt and a model
    list_input_text: list input text
    prompt: prompt
    service provide: either "openai" or "dglc" for the moment
    model: model name (as parameter of the query)
    temperature: temperature (0: precise, 1: creative)
    cache_prefix_fp: prefix of the file to which write content of cache after each call
    """

    if cache_prefix_fp is not None:
        cache_fp = cache_prefix_fp + "___" + "__".join([service_provider, model, str(temperature)]).replace(" ", "_") + ".json"

        if os.path.exists(cache_fp):
            with open(cache_fp) as f:
                map_query_input_output = json.load(f)
        else:
            map_query_input_output = {}
    else:
        map_query_input_output = None
        cache_fp = None

    handler = None
    #if service_provider.lower() == "dglc": handler = api_call_dglc
    if service_provider.lower() == "openai": handler = api_call_openai
    if service_provider.lower() == "gptjrc": handler = api_call_gptjrc
    if service_provider.lower() == "gptec": handler = api_call_gptec

    list_output_text = []
    for input_text in tqdm(list_input_text):
        output_text = call_model_with_caching(input_text, prompt, model, temperature, handler, map_query_input_output,
                                              cache_fp, delimiter=delimiter, InContextExamples=InContextExamples)
        list_output_text.append(output_text)

    return list_output_text




if __name__ == "__main__":

    USE_CACHE = False #True #False

    #service_provider = "openai"
    #model_name = "gpt-3.5-turbo-16k"
    #
    #
    #service_provider = "dglc"
    # dglc available models: 'OA_SFT_Pythia_12B', 'JRC_RHLF_13B', 'OA_GPT3.5', 'OA_GPT3'
    # model_name = "gpt-3.5-turbo"  #OpenAI name
    # model_name = 'JRC_RHLF_13B'
    #model_name = "OA_SFT_Pythia_12B"   #EleutherAI-pythia-12b
    # model_name = "OA_GPT3"
    # model_name = "GPT@JRC_4"
    #
    #
    #service_provider = "gptjrc"
    service_provider = "gptec"

    #model_name = "gpt-5.1"  #GPT-

    #model_name = "llama-3.3-70b-instruct"
    
    
    #model_name="mistral-small-3.2-24b"
    
    model_name="gpt-oss-120b"

    #model_name="minimax-m2"
    
    #model_name = "qwen3.5-27b-fp8"

    # temperature: temperature_value (0: precise, 1: creative)
    temperature_value = 0.01  # 0.1

    ##################################################################################################

    #OpenAI ChatGPT API
    if service_provider == "openai":
        MyOpenAPIKey = ""
        fkeyname="OpenAI-APItoken.key"
        if os.path.exists(fkeyname):
            with open(fkeyname) as f:
                MyOpenAPIKey = f.read().strip()
        setup_openai(key=MyOpenAPIKey)

    # # test api call
    # r = api_call_openai("say hello world", "you will answer in Spanish", "gpt-3.5-turbo", 0)
    # print(r)

    # # test process list
    # r = process_list(["hello world", "hello everybody"], "you will translate to Spanish", "openai", "gpt-3.5-turbo", 0)
    # print(r)

    # # process list with caching
    # r = process_list(["hello world", "hello everybody"], "you will translate to Spanish", "openai", "gpt-3.5-turbo", 0, "UNITTEST")
    # print(r)

    #### GPT@JRC API
    if service_provider == "gptjrc":
        key_gptjrc = ""
        fkeyname = "GPTJRC-APItoken.key"
        if os.path.exists(fkeyname):
            with open(fkeyname) as f:
                key_gptjrc = f.read().strip()
        setup_gptjrc(key_gptjrc)

    #### GPT@EC API
    if service_provider == "gptec":
        key_gptec = ""
        fkeyname = "GPTEC-APItoken.key"
        if os.path.exists(fkeyname):
            with open(fkeyname) as f:
                key_gptec = f.read().strip()
        else:
            key_gptec = os.environ.get('key_gptec', '')
        setup_gptec(key_gptec)

    #### DGLC API
    if service_provider == "dglc":
        key_dglc = ""
        fkeyname = "DGLC-APItoken.key"
        if os.path.exists(fkeyname):
            with open(fkeyname) as f:
                key_dglc = f.read().strip()
        # setup_dglc(key=key_dglc)

    # TEST OF DGLC API CALL
    # input_text = "this morning a rabbit killed a hunter"
    # print("\ntext = \n"+input_text)
    # prompt = "please reformulate the text, add more details, the text should be between 200 and 500 characters:"
    # print("\nquestion = \n" + prompt)
    # model= "OA_SFT_Pythia_12B" #not available in OpenAI
    # print("\nmodel = \n" + model)
    # print("\n")
    #
    # r = api_call_dglc(prompt, input_text, model, 0.5)
    # if r:
    #     print(r)
    #
    # print("\nDone!\n")


    ###########################################################################


    if USE_CACHE:
        # cache_prefix_fp: prefix of the file to which write content of cache after each call
        cache_prefix_fp = "LLMQUERYTEST"
        cache_name = cache_prefix_fp + "___" + "__".join([service_provider, model_name, str(temperature_value)]).replace(" ", "_") + ".json"

        if os.path.exists(cache_name):
            with open(cache_name) as f:
                load_map_query_input_output = json.load(f)
        else:
            load_map_query_input_output = {}

    myPromt = f"""
        translate in Spanish the text below, delimited by triple  \
        Text: 
        """

    myDelimiter = "```"

    ###

    #example for counting number of tokens of the overall prompt for the model

    # input_text = "one, two, three, a step fortward Mary"
    # overall_string = myPromt + myDelimiter + input_text + myDelimiter
    # nt = token_counter(overall_string, model_name)
    # print("\nNumber of Tokens in the example = "+str(nt))

    encod = encoding_getter(model_name)
    print("\nencodName = " + str(encod.name))

    InContextExamples = []
    # InContextExamples = [["ADENOVIRUS - USA (02): (NEW JERSEY) UPDATE A ProMED-mail post http://www.promedmail.org ProMED-mail is a program of the International Society for Infectious Diseases http://www.isid.org Date: Sun 28 Oct 2018 3:12 PM Source: CBS news [edited] https://www.cbsnews.com/news/adenovirus-outbreak-new-jersey-wanaque-center- facility-9th-child-dies-2018-10-28/ Another child has died amid a deadly virus outbreak at a New Jersey rehabilitation, bringing the total number of deaths to 9, officials said [Sun 28 Oct 2018]. The state\'s Department of Health said the victim was a \"medically fragile\" child who had a confirmed case of adenovirus. The unidentified child died Saturday evening [27 Oct 2018] at the Wanaque Center for Nursing and Rehabilitation in Haskell, the department said. There have been 25 cases associated with the outbreak. \"This is a tragic situation, and our thoughts are with the families who are grieving right now,\" Health Commissioner Dr Shereef Elnahal said in a statement. \"We are working every day to ensure all infection control protocols are continuously followed and closely monitoring the situation at the facility.\" Adenoviruses are a family of viruses that account for about 5 to 10 percent of fevers in young children, but most patients recover. The infections can affect the tissue linings of the respiratory tract, eyes, intestines, urinary tract and nervous system, causing illnesses ranging from a cold to bronchitis to pneumonia to pink eye. The children at Wanaque appear to have been more susceptible to serious infections due to their other medical conditions. Children at the center are severely disabled, with some living in comas, and for many, it is their permanent home, the Bergen Record reports. Many will never walk or talk, and some have spent virtually their whole lives there, according to the paper. . Communicated by: ProMED-mail Rapporteur Kunihiko Iizuka [Human adenoviruses (HAdVs) are non-enveloped, linear double-stranded DNA viruses encapsidated within a protein shell and have been categorized into 6 species (A-F) that contain 51 immunologically distinct serotypes (Fields virology. 5th ed. Philadelphia (PA): Lippincott-Raven; 2007. p. 2395-436). HAdVs most commonly cause acute respiratory disease; however, depending on the infecting HAdV serotype and tropism resulting from differential host receptor use, the wide variety of symptoms can include pneumonia, febrile upper respiratory illness, conjunctivitis, cystitis, and gastroenteritis (Principles and practice of infectious diseases. 5th ed. Vol 2. Philadelphia (PA): Churchill Livingstone; 2005. p. 1835-41). The severity of disease appears dependent on the immunocompetence and cardiopulmonary health of the host, and the spectrum of disease can range from subclinical to severe respiratory distress and death. Immunocompromised patients (such as bone marrow transplant recipients) are particularly susceptible to HAdV infection, resulting in severe illness and deaths, whereas illness in immunocompetent patients generally resolves without major complication. The outbreak report above involves young children that are in a healthcare facility and immunocompromised on account of underlying co- morbid conditions. Adenovirus associated morbidity and mortality in this setting would require intensive infection control measures. In immunocompromised patients, several drugs, such as cidofovir, ribavirin, ganciclovir, and vidarabine, have been used to treat adenovirus infections. Most of these agents are virostatic, may induce drug resistance, and have significant risks of toxicities, as well as risks to healthcare staff [e.g., aerosolized ribavirin - Mod.ML]. - Mod.UBA HealthMap/ProMED map available at: New Jersey, United States: https://promedmail.org/promed-post?place=6117463,232 ] See Also Adenovirus - USA: (NJ) children, fatal 20181025.6108470 .uba/mj/ml",
    #                          "{\"virus\": \"ADENOVIRUS\", \"country\": \"USA\", \"date\": \"2018-10-28\", \"cases\": \"25\", \"deaths\": \"9\"}"],
    #                      ["NIPAH VIRUS - INDIA (14): (KERALA) * A ProMED-mail post http://www.promedmail.org ProMED-mail is a program of the International Society for Infectious Diseases http://www.isid.org Date: Tue 3 Jul 2018 Source: MediBulletin [edited] https://medibulletin.com/2018/07/03/bats-indicted-in-kerala-nipah-outbreak- icmr-sends-paper-to-lancet/ Putting to rest suspense about the source of the Nipah virus infections in Kerala, scientists from the Indian Council of Medical Research have now found the virus in bats that were caught from the affected areas. At least 17 people died of Nipah infection in Mallapuram and Kozhikode districts of Kerala over April and May [2018]. While the 1st batch of bats caught from the well in Kozhikode in the house from where the 1st case was reported, had tested negative; of the 2nd batch of 52 fruit bats, 19.2 percent were found to carry the virus. The findings will be published in The Lancet. Health minister J P Nadda was informed about the findings in a meeting last week. In the meeting, scientists from ICMR and public health officials also told the minister that circumstances have now improved enough for the state to be declared Nipah free. The incubation period of Nipah is 5 to 14 days. The last case was in May [2018] and now that 2 incubation periods have elapsed without any fresh cases, the specter of the dreaded disease seems to be finally receding. . Communicated by: ProMED-mail <promed@promedmail.org> [It is good to learn that there have been no additional cases of Nipah virus infection in Kerala. As was mentioned earlier, it is not surprising that the bats taken from the well were negative for the virus. Giant fruit bats (flying foxes genus _Pteropus_), the reservoir of Nipah virus in Bangladesh and Malaysia, do not roost in wells. They roost in tree tops. The species sampled in the 2nd batch of bats was not mentioned, but were likely flying foxes. It is fortunate that virus positive bats were found in this 2nd sampling. As commented earlier, bats \"may only be infectious for a week or 2, and then they clear the virus and they\'re no longer infectious,\" said Jonathan Epstein, a veterinarian and epidemiologist at EcoHealth Alliance, New York, who has, for over a decade, studied Nipah outbreaks and the bats that cause them, in Malaysia, India and Bangladesh. \"That\'s why these outbreaks are relatively rare events, given the fact that these bats are so abundant and so common but very few of them are ever actually shedding virus at a given time.\" Epstein and others conducted an experimental study of _Pteropus_ bats in 2011 and found that the time window in which the bats are capable of passing on the infection to other animals or humans is quite small. In fact, the virus can\'t be found in experimentally infected bats after a few weeks. The few bats in an infected population that could be shedding the virus may be doing so in low quantities and for a short duration. \"Finding that bats don\'t have Nipah virus at the time of sampling certainly doesn\'t mean that it didn\'t come from those bats, particularly _P. medius_,\" Epstein said. \"The overwhelming abundance of evidence really shows that this bat is the reservoir for Nipah virus on the subcontinent in Bangladesh and in India.\" - Mod.TY Maps of India can be accessed at: http://www.mapsofindia.com/maps/india/india-political-map.htm HealthMap/ProMED map available at: Kerala State, India: https://promedmail.org/promed-post?place=5887151,308 ] See Also Nipah virus - India (13): (KL) 20180611.5851326 Nipah virus - India (12) 20180603.5836554 Nipah virus - India (11): (KL) 20180602.5835342 Nipah virus - India (10): (KL) 20180602.5833137 Nipah virus - India (09): (WB ex KL) susp. 20180530.5829184 Nipah virus - India (08): (KR ex KL) susp. 20180529.5826769 Nipah virus - India (07) 20180528.5822566 Nipah virus - India (06): (KL,KA) 20180526.5819777 Nipah virus - India (05): (KL,TG) 20180525.5817917 Nipah virus - India (04): (KL, KA) 20180524.5815473 Nipah virus - India (03): (KL) conf. 20180522.5812214 Nipah virus - India (02): (KL) conf 20180521.5809003 Nipah virus - India: (KL) susp 20180521.5807513 2007 . Nipah virus, fatal - India (West Bengal) (02) 20070511.1514 Nipah virus, fatal - India (West Bengal) 20070508.1484 Undiagnosed deaths - Bangladesh, India (04) 20070504.1451 .ty/ao/jh",
    #                          "{\"virus\": \"Nipah virus\", \"country\": \"India\", \"date\": \"2018-07-03\", \"cases\": \"None\", \"deaths\": \"17\"}"],
    #                      ["UNDIAGNOSED RESPIRATORY ILLNESS - USA: (NEW YORK), ex MIDDLE EAST, FLIGHT PASSENGERS AND CREW, REQUEST FOR INFORMATION * A ProMED-mail post http://www.promedmail.org ProMED-mail is a program of the International Society for Infectious Diseases http://www.isid.org [1] Date: Wed 5 Sep 2018 Source: Stat News [edited] https://www.statnews.com/2018/09/05/plane-quarantined-at-kennedy-airport-amid- reports-of-100-ill-passengers/ An Emirates Airline flight was held in quarantine for a period at New York\'s John F. Kennedy International Airport after a large number of passengers were reported feeling ill during the flight. The airline said in a statement that about 10 passengers on the flight from Dubai to New York had become ill on board. But the Centers for Disease Control and Prevention [CDC] said the estimated number was about 100. Passengers and some crew complained of illness including cough, according to the CDC; some had developed a fever. \"CDC public health officers are working with port authority, EMS, and CBP officials to evaluate passengers including taking temperatures and making arrangements for transport to local hospitals,\" the CDC said. \"Passengers who are not ill will be allowed to continue with their travel plans, and if necessary will be followed up with by health officials.\" A spokesman for New York Mayor Bill de Blasio said 19 passengers had taken ill - 10 were sent to a hospital and another 9 refused medical attention. There were approximately 521 passengers on the flight. A number of the passengers on the flight were returning from the Hajj, the annual mass pilgrimage to Mecca, in Saudi Arabia, a source told STAT. It was unclear if the people who were ill were the same passengers who had attended. Saudi Arabia has reported cases of MERS, Middle East respiratory syndrome, which passes to people from camels. But the fact that so many people became ill during the flight would make MERS seem an unlikely cause. [Byline: Helen Branswell] . Communicated by: Meghan May University of New England <mmay3@une.edu> [2] Date: Wed 5 Sep 2018 Source: BBC [edited] https://www.bbc.com/news/world-us-canada-45425412 A total of 19 people have been taken ill after an Emirates airline plane landed in New York, officials say. The plane was quarantined at JFK airport as those on board were checked by health officials. As many as 10 were taken to hospital but others refused treatment. The US Centers for Disease Control and Prevention (CDC) said that initially about 100 people including some crew had complained of illness. Flight 203 from Dubai landed at 09:10 (13.10 GMT) with 521 passengers. Emergency vehicles were seen on the runway as it landed. Soon afterwards, Emirates airline tweeted that the sick passengers were being attended to and those who were unaffected would be allowed to leave the plane. The CDC said in a statement that is was \"aware of an Emirates flight from Dubai that arrived this morning at JFK\". \"Approximately 100 passengers, including some crew on the flight, complained of illness including cough and some with fever. \"CDC public health officers are working with. officials to evaluate passengers including taking temperatures and making arrangements for transport to local hospitals those that need care.\" Later Eric Phillips, spokesman for New York Mayor Bill de Blasio, confirmed that all the passengers were off the plane and the sick people had been taken to hospital. He said that some of the passengers had originally come from the Saudi Arabian city of Mecca, which was currently experiencing a flu outbreak, and that the passengers\' symptoms were \"pointing to the flu\". . Communicated by: ProMED-mail <promed@promedmail.org> [3] Date: Wed 5 Sep 2018, 10:55 AM ET Source: NPR [edited] https://www.npr.org/2018/09/05/644826743/emirates-airline-says- about-10-passengers-fell-ill-on-flight-to-new-york Health and safety officials are investigating an illness that struck people on an Emirates Airline flight from Dubai, United Arab Emirates, to New York\'s John F. Kennedy International Airport on Wednesday morning. A total of 7 crew members and 3 passengers were taken to the hospital, Emirates Airline said. It added that [Wed 5 Sep 2018] return flight from New York to Dubai would leave 3 hours late. The Centers for Disease Control and Prevention said around 100 people on the overnight Flight 203 had complained of illness. For some, the symptoms were limited to a cough; others had a fever. An Emirates A380 in quarantine at JFK Airport right now awaiting CDC officials after about 100 passengers became ill with fevers over 100 degrees and coughing. Flight 203 had just arrived from Dubai. Ten people were taken off the plane for treatment at Jamaica Hospital Medical Center, said Eric Phillips, press secretary for New York City Mayor Bill de Blasio. He said 9 others were found to be sick but refused medical attention. The aircraft was carrying around 521 passengers. Health officials allowed people to disembark only after checking each one for symptoms, Phillips stated, adding, \"The plane\'s been quarantined and the CDC is on the scene.\" As for what the sickness might be, Phillips referred to a \"flu outbreak\" in Mecca, Saudi Arabia, and said that might be a possibility, stating, \"It appears some of the ill passengers came from Mecca before getting on in Dubai.\" [One] passenger called it the \"worst flight ever,\" saying on Twitter that the plane \"was basically a flying infirmary. Many of these people should never have been allowed to board.\" By around noon, 432 passengers had been cleared and allowed to go to the customs area, according to Phillips. A few others who showed symptoms were held for treatment and possible transport to the hospital. NPR\'s Rob Stein reports that the Centers for Disease Control and Prevention is working with local authorities. The Airbus A380 jet was isolated on the tarmac at JFK, as officials took stock of the situation. Images from the scene showed a row of ambulances alongside the aircraft. [Another] passenger said via Twitter that CDC staff came onto the plane and that everyone aboard was asked to fill out a form providing their contact information for the next 3 weeks. [Byline: Bill Chappell] . Communicated by: ProMED-mail <promed@promedmail.org> [Mention a plane load of individuals with febrile respiratory symptoms coming from the Middle East and immediate panic of \"could this be MERS-CoV?\" Equally or more serious than this is an outbreak of influenza. According to the media coverage, there is an ongoing outbreak of influenza in Mecca (Makkah) (Saudi Arabia, where Hajji\'s congregate) now, concurrent with the Hajj, and a rapid onset of respiratory symptoms is more likely to be influenza than it is MERS- CoV. We still do not have information as to the origins of the ill passengers to know if they were beginning their trips in Mecca and were returning Hajji\'s or if they began their travels elsewhere. Presumably the crew members began their working journeys in Dubai, but may have commuted in from elsewhere in or outside of the region (and there is mention of ill crew members). Returning to the possibility that this is related to the Hajj, a mass gathering of more than 2 million individuals from all over the world, making a religious pilgrimage to Mecca, it is interesting to review the Saudi Hajj/Umrah health requirements. While some vaccines are mandatory (meningitis vaccine, polio if coming from a country with ongoing poliovirus transmission, and yellow fever if coming from a known yellow fever endemic area, in contrast, influenza vaccine is recommended but not obligatory. \"Seasonal Influenza: \"The Saudi Ministry of Health recommends that international pilgrims be vaccinated against seasonal influenza before arrival into the Kingdom of Saudi Arabia, particularly those at increased risk of severe influenza diseases, including pregnant women, children under 5 years, the elderly, and individuals with underlying health conditions such as HIV/AIDS, asthma, and chronic heart or lung diseases. In Saudi Arabia, seasonal influenza vaccination is recommended for internal pilgrims, particularly those at risk described above, and all health-care workers in the Hajj premises.\" https://www.saudiembassy.net/hajj-and-umrah-health-requirements Given concerns re: possible MERS-CoV transmission to pilgrims visiting Saudi Arabia, many countries have increased surveillance of respiratory illnesses in returning pilgrims (notably post Hajj), and in the 6 years since identification of the MERS-CoV, there have been no cases among returning Hajji\'s and just a handful of cases among individuals returning from having performed the Umrah pilgrimage. Movement and exposures of visiting Hajj pilgrims are controlled - camels are not permitted in the area where Hajjis are congregated, healthcare workers and others identified as contacts of confirmed MERS-CoV cases are not permitted to go to the Hajj area Studies addressing the etiologies of respiratory illnesses in returning Hajji\'s have identified influenza virus, respiratory syncytial virus, parainfluenza virus, adenovirus and rhinovirus (see Respiratory infections - UK: Hajj returnees 20151011.3706464 and Respiratory virus infections - Saudi Arabia: Hajj pilgrims 2012 20130730.1854631 ). Note that the incubation period for influenza ranges from 1-4 days so transmission during travel is a plausible event. The actual number of individual identified as ill enough to require medical treatment at a hospital seems to be between 19 and 27 (taking into account the additional 8 identified and mentioned by the New York City\'s mayor\'s spokesperson in a tweet). Presumably these individuals had a fever above 100 F (37.8 C) and cough. The extension of the count to approaching 100 individuals may include those with a cough, possibly in the early stages of illness (influenza frequently begins with a dry cough), and others possibly coughing in response to hearing others coughing around them (think theater or lectures when coughing begins.) As the rapid respiratory virus screening tests should be available in a reasonable amount of time, we will post information as it becomes available, but should any knowledgeable sources have additional information that can be shared with us, ProMED would be very grateful. The HealthMap/ProMED maps can be found at: New York State, United States: https://promedmail.org/promed-post?place=6009759,234 Middle East: https://promedmail.org/promed-post?place=6009759,12214 . Mod.MPP] See Also MERS-CoV (29): UK (England) ex Saudi Arabia, Risk Assessment ECDC 20180830.5996187 2017 . MERS-CoV (63): Saudi Arabia (QS, TB), WHO : 20171009.5369268 2016 . Health hazards - Saudi Arabia: Hajj, Umrah, vaccination requirements 20160715.4346367 MERS-CoV (71): Saudi Arabia (MK), pilgrimage caution, WHO 20160623.4305152 2015 . MERS-CoV (138): Saudi Arabia, Jordan, post Hajj surveillance, RFI 20151009.3704734 MERS-CoV (136): Kuwait WHO, Saudi Arabia MOH, camel, Hajj 20150924.3666811 MERS-CoV (131): Saudi Arabia, animal reservoir, camels, Hajj, RFI 20150914.3643612 MERS-CoV (130): Saudi Arabia, animal reservoir, camels, Hajj 20150912.3641457 MERS-CoV (114): Saudi Arabia, animal reservoir, camels, Hajj 20150823.3597358 Respiratory infections - UK: Hajj returnees 20151011.3706464 Respiratory infections - Canada: (AB) Hajj returnees 20151020.3729641 Influenza (51): Germany ex Saudi Arabia, Hajj returnee, RFI 20151009.3704297 Influenza (49): Canada ex Saudi Arabia, Hajj returnees, susp., RFI, Alert 20151005.3693052 2014 . Meningitis, meningococcal - Saudi Arabia: prevention, Hajj travel advice 20140815.2692227 2013 . Respiratory virus infections - Saudi Arabia: Hajj pilgrims 2012 20130730.1854631 2012 . Health hazards - Saudi Arabia: updated Hajj advice 20121011.1338172 2009 . Influenza pandemic (H1N1) 2009 (113): Saudi Arabia, Hajj fatalities 20091122.4013 Influenza pandemic (H1N1) 2009 (109): Saudi Arabia, Hajj pilgrims 20091120.3997 2006 . Influenza - Saudi Arabia: Hajj concerns 20061209.3478 .mpp/ao/mpp",
    #                          "{\"virus\": \"None\", \"country\": \"United Arab Emirates\", \"date\": \"2018-09-05\", \"cases\": \"19\", \"deaths\": \"None\"}"]]

    if InContextExamples:
        ntotExamplesTokens = 0
        for row in InContextExamples:
            for col in row:
                nt = token_counter(col, model_name)
                #print("\nNumber of Tokens in the example = " + str(nt))
                ntotExamplesTokens = ntotExamplesTokens + nt
        #
        print("\nNumber of Tokens of the all examples = " + str(ntotExamplesTokens))

    ###

    if service_provider == "openai":
        if USE_CACHE:
            lambda_model = partial(call_model_with_caching, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter, InContextExamples=InContextExamples, handler=api_call_openai,
                                   map_query_input_output=load_map_query_input_output, cache_fp=cache_name, verbose=True)
        else:
            lambda_model = partial(call_model, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter, InContextExamples=InContextExamples, handler=api_call_openai,
                                   verbose=True)
    elif service_provider == "gptjrc":
        if USE_CACHE:
            lambda_model = partial(call_model_with_caching, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter, InContextExamples=InContextExamples, handler=api_call_gptjrc,
                                   map_query_input_output=load_map_query_input_output, cache_fp=cache_name, verbose=True)
        else:
            lambda_model = partial(call_model, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter, InContextExamples=InContextExamples, handler=api_call_gptjrc,
                                   verbose=True)
    elif service_provider == "gptec":
        if USE_CACHE:
            lambda_model = partial(call_model_with_caching, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter, InContextExamples=InContextExamples, handler=api_call_gptec,
                                   map_query_input_output=load_map_query_input_output, cache_fp=cache_name, verbose=True)
        else:
            lambda_model = partial(call_model, prompt=myPromt, model=model_name,
                                   temperature=temperature_value, delimiter=myDelimiter, InContextExamples=InContextExamples, handler=api_call_gptec,
                                   verbose=True)
    # elif service_provider == "dglc":
    #     if USE_CACHE:
    #         lambda_model = partial(call_model_with_caching, prompt=myPromt, model=model_name, temperature=temperature_value, delimiter=myDelimiter, InContextExamples=InContextExamples,
    #                                handler=api_call_dglc, map_query_input_output=load_map_query_input_output, cache_fp=cache_name, verbose=True)
    #     else:
    #         lambda_model = partial(call_model, prompt=myPromt, model=model_name,
    #                                temperature=temperature_value, delimiter=myDelimiter, InContextExamples=InContextExamples, handler=api_call_dglc,
    #                                verbose=True)


    if lambda_model:
        df = pd.DataFrame([["one, two, three, a step fortward Mary"], ["one, two, three, a step back"]], columns=["text"])
        df["text_es"] = df["text"].apply(lambda_model)

        print("\n")
        print(df)

    print("\nEnd Computations")



