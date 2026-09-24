from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from guardrails import prompting


def build_prompt(text: str) -> str:
    """Prompt de inferencia. Debe coincidir con el del entrenamiento."""
    return prompting.build_prompt(text)


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return {"decision": "ERROR", "primary_label": "PARSE_ERROR", "labels": ["PARSE_ERROR"]}
    return json.loads(match.group(0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="models/guardrail-qwen25-1_5b-qlora-v1")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter if Path(args.adapter).exists() else args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    print(json.dumps({"event": "ready"}), file=sys.stderr, flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            text = str(request["text"])
            started = time.perf_counter()
            prompt = build_prompt(text)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            parsed = extract_json(generated)
            response = {
                "id": request["id"],
                "text": text,
                "raw": generated.strip(),
                "parsed": parsed,
                "decision": parsed.get("decision", "ERROR"),
                "primary_label": parsed.get("primary_label", "UNKNOWN"),
                "labels": parsed.get("labels", []),
                "model_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            response = {"id": request.get("id") if "request" in locals() else None, "error": repr(exc)}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
