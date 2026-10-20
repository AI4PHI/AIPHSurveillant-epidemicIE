"""
Fine-tuned extraction model inference helpers (local, open-weight inference).

Prompt building, model loading and generation for the fine-tuned extraction model
(jattokatarratto/EpiQwen3.5-9B, a LoRA fine-tune served locally) and its legacy predecessors.
The daily pipeline (llm_extraction_AutomatedDynamicPipeline.py) imports build_prompt(),
load_causal_lm() and generate_response() from this module via importlib (the file name contains
hyphens), and reuses them as-is for the local fine-tuned-model extraction path.

It can also be run standalone as a CLI to extract disease-outbreak information from a single text:

    python causal_LLM_inference-Run-EPIextraction.py --input_text "..."

Chat-format fine-tunes (e.g. EpiQwen3.5-9B) ship a serving_prompt_format.json and are served with
the chat template they were trained on; older ones (e.g. AI4PH/EpiQwen2.5-7B) use the Alpaca
"### Instruction" prompt built by build_prompt(). serving_format.find_serving_format() tells the
two apart.
"""
# %%
import os

# %%
# Optional runtime configuration (uncomment and adapt to your environment):
# os.environ["HF_HUB_CACHE"] = "/path/to/huggingface/hub"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/path/to/huggingface/hub"
# os.environ["HF_HOME"] = "/path/to/huggingface/hub"
#
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
import torch
import argparse
import time
import random
import numpy as np
import gc
import warnings

from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# serving-side counterpart of the chat-format training scripts: the fine-tunes trained on the chat
# template of their base model (jattokatarratto/EpiQwen3.5-9B) ship a serving_prompt_format.json and
# must be served with that very prompt and stop token, while the older ones (AI4PH/EpiQwen2.5-7B)
# keep the Alpaca prompt of build_prompt() below. find_serving_format() tells the two apart.
import serving_format

warnings.filterwarnings("ignore", message=".*use_reentrant parameter should be passed explicitly.*")


def clean_input_text(text):
    """Clean input text by normalizing newlines and whitespace."""
    if text:
        text = text.replace("\\n\\n", ". ")
        text = text.replace("\n\n", ". ")
        text = text.replace("\r\n", " ")
        text = text.replace("\r", " ")
        text = text.replace("\\n", " ")
        text = text.replace("\n", " ")
        while "  " in text:
            text = text.replace("  ", " ")
    return text


def build_prompt(input_text):
    """Build the extraction prompt for a given input text."""

    instruction = (
        "Extract disease outbreak information from the text into JSON format.\n"
        "Return a list containing one JSON object per outbreak mentioned.\n"
        "Always return a list of JSON objects, even for single outbreaks.\n"
        "Use NaN for missing information.\n"
        "If no outbreak information is found, return an empty list [].\n"
        "Do not invent information not present in the text."
    )

    base_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            {}

            ### Input:
            {}

            ### Response:
            {}"""

    cleaned_text = clean_input_text(input_text)
    prompt = base_prompt.format(instruction, cleaned_text, "")
    return prompt


def load_causal_lm(model_id, fmt, quantization_config=None, **kwargs):
    """
    Legacy fine-tunes: AutoModelForCausalLM as before. Chat-format fine-tunes (fmt not None): the class
    used in training (config.architectures; for Qwen3.5 AutoModelForCausalLM is a different class whose
    parameter names don't match the adapter) and sdpa attention (Gemma 4 global layers: head_dim 512).
    """
    if fmt is None:
        return AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config,
                                                    torch_dtype=torch.bfloat16, **kwargs)
    model_cls = serving_format.load_model_class(model_id)
    return model_cls.from_pretrained(model_id, quantization_config=quantization_config,
                                     dtype=torch.bfloat16, attn_implementation="sdpa", **kwargs)


def generate_response(model, tokenizer, input_text, max_new_tokens=600, fmt=None):
    """Generate a response from the model given a prompt."""

    if fmt is not None:
        # chat-format fine-tune: prompt already holds its special tokens, greedy, chat stop token
        inputs = serving_format.tokenize_prompts(tokenizer, [input_text], device=model.device)
        input_length = inputs['input_ids'].shape[1]
        print(f"Input prompt length: {input_length} tokens")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     **serving_format.greedy_kwargs(tokenizer, fmt))
        return serving_format.decode_answer(tokenizer, outputs[0][input_length:], fmt)

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=False,
        padding=False,
    ).to(model.device)

    input_length = inputs['input_ids'].shape[1]
    print(f"Input prompt length: {input_length} tokens")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run EPI extraction inference with a fine-tuned model.")

    parser.add_argument("--load_from", type=str, default="huggingface",
                        choices=["huggingface", "disk"],
                        help="Load the model from 'huggingface' (--hf_model_id) or 'disk' (--model_path). "
                             "Either may be a PEFT adapter (has adapter_config.json: loaded on its base "
                             "model) or a full / merged model (loaded directly)")
    # the fine-tuned extraction model of the pipeline; its predecessor was "AI4PH/EpiQwen2.5-7B"
    parser.add_argument("--hf_model_id", type=str,
                        default="jattokatarratto/EpiQwen3.5-9B",
                        help="HuggingFace repo id, adapter or merged model (used when --load_from "
                             "huggingface). Chat-format fine-tunes need serving_prompt_format.json "
                             "in the repo")
    parser.add_argument("--model_path", type=str,
                        default="",
                        help="Fine-tuned model on disk (used when --load_from disk): a merged model or "
                             "an adapter directory")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model for a PEFT adapter. Default: model_id in "
                             "serving_prompt_format.json, else base_model_name_or_path in "
                             "adapter_config.json (e.g. Qwen/Qwen3.5-9B)")
    parser.add_argument("--max_new_tokens", type=int, default=600,
                        help="Maximum number of new tokens to generate (chat-format fine-tunes: the "
                             "value recommended by their serving_prompt_format.json)")
    parser.add_argument("--input_text", type=str, default=None,
                        help="Input text for extraction (if not provided, a default example is used)")
    parser.add_argument("--load_4bit", type=str, default="auto",
                        help="auto: 4-bit for legacy Alpaca-format fine-tunes (as before), bf16 for "
                             "chat-format fine-tunes (with serving_prompt_format.json). Or True / False.")
    parser.add_argument("--SEED", type=int, default=41)

    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)

    # Default example text
    if args.input_text is None:
        args.input_text = """22 January 1996  

Disease Outbreak Reported

During the past 10 years the incidence of meningococcal meningitis increased
gradually in Rio de Janeiro - from 290 cases (2.4 per 100.000 population) in
1986 to 853 (8.3 per 100,000) in 1993. Preliminary data for 1994 are 974 cases
and for 1995, 1,106 notified by 28 December 1995. In 1995, 40% of the cases
were in children under 5 and 16% of the cases died. Most cases (90%) in 1995
were from the metropolitan area and about half from Rio de Janeiro. Types B
and C have increased during this period and in 1995 when 48% of cases were
serotyped, 42% were type B and 57% type C. Ampicillin and penicillin has been
recommended for treatment of cases, rifampicin for prophylaxis in close
contacts. Several vaccination campaigns have been conducted in the state, In
December 1994- February 1995, 63% of children in age groups from 6 months to
13 years. Over 80% in age groups between 1 and 9 years received the vaccine.
The vaccination campaign was resumed in November 1995 targeting the age groups
14-29 years."""

    print("=" * 70)
    print("EPI EXTRACTION - Fine-Tuned Model Inference")
    print("=" * 70)

    start = time.time()

    # Chat-format fine-tunes (EpiQwen3.5-9B) ship serving_prompt_format.json and are served with the
    # chat template they were trained on; older ones (EpiQwen2.5-7B) use the Alpaca prompt above.
    model_src = args.model_path if args.load_from == "disk" else args.hf_model_id
    fmt = serving_format.find_serving_format(model_src)
    adapter_base = serving_format.adapter_base_model(model_src)   # None: full / merged model
    load_4bit = (fmt is None) if args.load_4bit == "auto" else args.load_4bit.lower() in ("true", "1", "yes")
    if fmt is not None and args.max_new_tokens == parser.get_default("max_new_tokens"):
        # generation stops on the chat stop token, so the trained answer budget costs nothing
        args.max_new_tokens = fmt.get("recommended_generation", {}).get("max_new_tokens", args.max_new_tokens)
    if adapter_base is not None and args.base_model is None:
        # the adapter must go on the model it was trained on, which the format file records
        args.base_model = fmt["model_id"] if fmt is not None else adapter_base
    print(f"Prompt format: {'chat template (' + fmt['model_id'] + ')' if fmt else 'Alpaca ### Instruction (legacy)'}"
          f" - weights in {'4-bit' if load_4bit else 'bf16'} - max_new_tokens {args.max_new_tokens}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Load model and tokenizer ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Tokenizer: legacy adapters use the base model's (as before); chat-format fine-tunes and full
    # models ship their own (chat template included).
    tokenizer_src = args.base_model if (adapter_base is not None and fmt is None) else model_src
    print(f"\nLoading tokenizer from: {tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if fmt is None:
        tokenizer.pad_token = tokenizer.eos_token  # chat-format fine-tunes keep their own pad
    tokenizer.padding_side = "left"

    weights_src = args.base_model if adapter_base is not None else model_src
    print(f"Loading {'base ' if adapter_base is not None else ''}model: {weights_src}")
    model = load_causal_lm(
        weights_src, fmt,
        quantization_config=bnb_config if load_4bit else None,
        device_map="auto",
        use_safetensors=True,
        trust_remote_code=True,
    )

    if adapter_base is not None:
        print(f"Loading PEFT adapter from {'HuggingFace' if args.load_from == 'huggingface' else 'disk'}: {model_src}")
        model = PeftModel.from_pretrained(model, model_src, torch_dtype=torch.bfloat16)
        if fmt is not None and not load_4bit:
            model = model.merge_and_unload()   # same weights as a merged model, faster generation
    model.eval()

    print("\nModel loaded successfully!\n")

    # Build prompt and run inference
    print("-" * 70)
    print("INPUT TEXT:")
    print("-" * 70)
    print(args.input_text[:500] + ("..." if len(args.input_text) > 500 else ""))
    print("-" * 70)

    prompt = build_prompt(args.input_text) if fmt is None else \
        serving_format.build_chat_prompt(tokenizer, fmt, args.input_text)

    print("\nGenerating extraction...")
    response = generate_response(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, fmt=fmt)

    print("\n" + "=" * 70)
    print("EXTRACTED OUTPUT:")
    print("=" * 70)
    print(response)
    print("=" * 70)

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nComputational Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
