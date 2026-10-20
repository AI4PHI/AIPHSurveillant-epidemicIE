"""
Serving-side counterpart of the chat-format training scripts
(causal_LLM_train.Parallel_hf_Qwen35.py, causal_LLM_train.Parallel_hf_Gemma4-it.py).

Those scripts train on the model's own chat template with thinking disabled and write the exact
format to serving_prompt_format.json. A model trained that way must be served with the SAME prompt
and must stop on the chat stop token (<|im_end|> for Qwen3.5, <turn|> for Gemma 4), otherwise it
sees a prompt it was never trained on and/or never stops.

Older fine-tunes (Qwen2.5-7B, Mistral, Llama-3.3 via causal_LLM_train.Parallel_hf_PreviousModels.py) have no
serving_prompt_format.json: find_serving_format() returns None for them and the callers keep the
original Alpaca "### Instruction" prompt, so their results are reproduced unchanged.
"""
import json
import os

FORMAT_FILE = "serving_prompt_format.json"

# End-of-thought markers of the empty thought block the templates put in the prompt. If a
# fine-tune re-emits the tail of that block ("thought\n<channel|>"), everything up to and including
# the LAST marker is dropped.
THOUGHT_END_TOKENS = ("<channel|>", "</think>")


def clean_input_text(input_text):
    """Same normalisation as clean_input_text() in the training scripts."""
    if not input_text or not isinstance(input_text, str):
        return ""
    input_text = input_text.replace("\\n\\n", ". ")
    input_text = input_text.replace("\n\n", ". ")
    input_text = input_text.replace("\r\n", " ")
    input_text = input_text.replace("\r", " ")
    input_text = input_text.replace("\\n", " ")
    input_text = input_text.replace("\n", " ")
    while "  " in input_text:
        input_text = input_text.replace("  ", " ")
    return input_text


def find_serving_format(model_path, verbose=True):
    """
    Look for serving_prompt_format.json for a model given as a local directory or a HF hub repo id.

    Local: the training scripts write it to dir_out_name/ (= .../hf/), the parent of saved-model/,
    merged-model/ and best-training-loss-model/, so the directory itself and its parent are searched.
    Hub: the file is downloaded if the repo has it.
    Returns the parsed dict, or None (= legacy Alpaca-format model).
    """
    if not model_path:
        return None
    if os.path.isdir(model_path):
        d = os.path.abspath(model_path)
        for cand in (os.path.join(d, FORMAT_FILE), os.path.join(os.path.dirname(d), FORMAT_FILE)):
            if os.path.isfile(cand):
                if verbose:
                    print(f"[serving-format] using {cand}")
                with open(cand) as fh:
                    return json.load(fh)
        return None
    try:
        from huggingface_hub import hf_hub_download
        cand = hf_hub_download(model_path, FORMAT_FILE)
    except Exception:
        return None
    if verbose:
        print(f"[serving-format] using {FORMAT_FILE} from hub repo {model_path}")
    with open(cand) as fh:
        return json.load(fh)


def adapter_base_model(model_path):
    """
    If model_path (local directory or HF hub repo id) is a PEFT adapter, return the base model it was
    trained on (base_model_name_or_path in adapter_config.json); None for a full / merged model.
    """
    if os.path.isdir(model_path):
        cfg = os.path.join(model_path, "adapter_config.json")
        if not os.path.isfile(cfg):
            return None
    else:
        from huggingface_hub import file_exists, hf_hub_download
        if not file_exists(model_path, "adapter_config.json"):
            return None
        cfg = hf_hub_download(model_path, "adapter_config.json")
    with open(cfg) as fh:
        return json.load(fh).get("base_model_name_or_path")


def load_model_class(model_id):
    """
    The class in config.architectures. For Qwen3.5 AutoModelForCausalLM resolves to a DIFFERENT
    (text-only) class than the one used for training, whose parameter names do not match the LoRA
    adapter; the training scripts therefore load the explicit class and so must inference.
    """
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    arch = (config.architectures or [None])[0]
    cls = getattr(transformers, arch, None) if arch else None
    if cls is None:
        print(f"[serving-format] WARNING: class '{arch}' not in transformers, "
              f"falling back to AutoModelForCausalLM")
        return AutoModelForCausalLM
    return cls


def build_chat_prompt(tokenizer, fmt, input_text, clean=True):
    """Rendered serve-time prompt (string), identical to render_prompt() in the training script."""
    if clean:
        input_text = clean_input_text(input_text)
    user = fmt["user_message_template"].format(instruction=fmt["instruction"], input_text=input_text)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        **fmt.get("chat_template_kwargs", {"enable_thinking": False}),
    )


def tokenize_prompts(tokenizer, prompts, device=None):
    """
    The rendered prompt already contains every special token (Gemma 4 puts <bos> in the text), so
    add_special_tokens=False; otherwise Gemma gets a double <bos>.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    return enc.to(device) if device is not None else enc


def stop_token_ids(tokenizer, fmt):
    ids = set()
    if fmt.get("eos_token_ids"):                      # Gemma 4: [<eos>, <turn|>]
        ids.update(int(i) for i in fmt["eos_token_ids"])
    if fmt.get("eos_token_id") is not None:           # Qwen3.5: <|im_end|>
        ids.add(int(fmt["eos_token_id"]))
    for s in fmt.get("stop_strings", []) + ([fmt["stop_token"]] if fmt.get("stop_token") else []):
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid != tokenizer.unk_token_id:
            ids.add(tid)
    if not ids:
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)


def pad_token_id(tokenizer, stop_ids):
    """The tokenizer's own pad token (Qwen3.5 <|endoftext|>, Gemma <pad>), else the first stop id."""
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stop_ids[0]


def greedy_kwargs(tokenizer, fmt):
    """
    generate() kwargs: greedy (the base generation_config of Gemma 4 samples with temperature 1.0,
    so do_sample=False must be explicit when an adapter is put on the base model) and the chat stop
    tokens.
    """
    stop_ids = stop_token_ids(tokenizer, fmt)
    return dict(do_sample=False, temperature=None, top_p=None, top_k=None,
                num_return_sequences=1,
                eos_token_id=stop_ids, pad_token_id=pad_token_id(tokenizer, stop_ids))


def decode_answer(tokenizer, generated_ids, fmt):
    """
    Generated ids (prompt removed) -> answer text: cut at the first stop token, drop any re-emitted
    empty thought block, then strip special tokens.
    """
    ids = [int(t) for t in generated_ids]
    stops = set(stop_token_ids(tokenizer, fmt))
    for k, t in enumerate(ids):
        if t in stops:
            ids = ids[:k]
            break
    for marker in THOUGHT_END_TOKENS:
        mid = tokenizer.convert_tokens_to_ids(marker)
        if mid is None or mid == tokenizer.unk_token_id:
            continue
        if mid in ids:
            ids = ids[len(ids) - ids[::-1].index(mid):]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()
