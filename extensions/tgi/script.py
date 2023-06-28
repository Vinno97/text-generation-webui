import json
from queue import Queue, Empty
import re
from pathlib import Path
from threading import Thread
from typing import Any, Iterable, Iterator

from langchain.llms.loading import load_llm
from langchain.llms.base import BaseLLM
from langchain.callbacks.aim_callback import AimCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from transformers import AutoTokenizer

from modules import shared
from modules.logging_colors import logger


def is_tgi_model(model_name: str):
    return model_name.startswith("tgi-")


def wrap_load_model(func, args, kwargs):
    """Wraps modules.models.load_model to add LangChain loader

    Ideally, we would add LangChain as a supported loader to the `load_func_map`, but this is not
    easily reachable from outside the function itself.
    """
    model_name: str = args[0]

    if not is_tgi_model(model_name):
        return func(*args, **kwargs)

    try:
        import text_generation
    except ImportError:
        raise ImportError(
            "Could not import text_generation python package. "
            "Please install it with `pip install text_generation`."
        )

    path_to_model = Path(f"{shared.args.model_dir}/{model_name}")

    llm_definition = path_to_model / "connection.json"

    if not llm_definition.exists():
        logger.error("No tgi-connection.json found for TGI model. Exiting.")
        return None, None

    model_config = json.loads(llm_definition.read_text())
    model_id = model_config["model_id"]

    model = text_generation.Client(
        model_config["inference_server_url"],
        timeout=model_config.get("timeout", 120),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer


def clean_stop_sequences(
    generated_text: str, stopping_strings: Iterable[str]
) -> tuple[str, bool]:
    is_truncated = False

    for stop_seq in stopping_strings:
        if stop_seq in generated_text:
            generated_text = generated_text[: generated_text.index(stop_seq)]
            is_truncated = True

    return generated_text, is_truncated

def wrap_generate_reply(func, args, kwargs):
    """Wraps modules.models.generate_reply to override generation behaviour for LangChain

    Ideally, we would add LangChain as a supported loader to the `load_func_map`, but this is not
    easily reachable from outside the function itself.
    """
    if not is_tgi_model(shared.model_name):
        return func(*args, **kwargs)

    import text_generation

    question = args[0]
    state = args[1]
    stopping_strings = args[2] or []

    model: text_generation.Client = shared.model


    seed = state["seed"] if state["seed"] > 0 else None
    typical_p = state["typical_p"] if state["typical_p"] < 1 else None
    top_p = state["top_p"] if state["top_p"] < 1 else None

    generate_params = dict(
        prompt=question,
        stop_sequences=stopping_strings,
        max_new_tokens=state["max_new_tokens"],
        top_k=state["top_k"],
        top_p=top_p,
        typical_p=typical_p,
        temperature=state["temperature"],
        repetition_penalty=state["repetition_penalty"],
        seed=seed,
        do_sample=state["do_sample"],
    )

    if shared.args.no_stream:
        res = model.generate(**generate_params)
        generated_text = res.generated_text

        # remove stop sequences from the end of the generated text
        generated_text, _ = clean_stop_sequences(generated_text)
        yield generated_text

    else:
        generated_text = ""
        for res in model.generate_stream(**generate_params):
            token = res.token
            if not token.special:
                generated_text += token.text

            generated_text, is_stopped = clean_stop_sequences(generated_text, stopping_strings)

            yield generated_text

            if is_stopped:
                break

