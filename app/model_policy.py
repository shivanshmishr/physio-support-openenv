from __future__ import annotations

import os
from functools import lru_cache

from app.heuristic_policy import heuristic_decision
from app.prompting import build_messages, render_plain_prompt
from app.structured_output import extract_json_object, validate_submission_payload


class HuggingFaceModelPolicy:
    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        allow_heuristic_fallback: bool = False,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.allow_heuristic_fallback = allow_heuristic_fallback
        self.tokenizer, self.model, self.device = self._load_model()

    def predict(self, observation: dict) -> dict:
        try:
            raw_text = self._generate(observation)
            payload = extract_json_object(raw_text)
            return validate_submission_payload(payload, observation)
        except Exception:
            if self.allow_heuristic_fallback:
                return heuristic_decision(observation)
            raise

    def _load_model(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for local model inference. "
                "Install requirements with transformers/torch/peft first."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            model_kwargs["dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(self.base_model, **model_kwargs)
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("peft is required to load a LoRA adapter checkpoint.") from exc
            model = PeftModel.from_pretrained(model, self.adapter_path)

        if not torch.cuda.is_available():
            model.to("cpu")

        model.eval()
        device = next(model.parameters()).device
        return tokenizer, model, device

    def _generate(self, observation: dict) -> str:
        import torch

        messages = build_messages(observation)
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            if getattr(self.tokenizer, "chat_template", None):
                chat_inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if hasattr(chat_inputs, "to"):
                    chat_inputs = chat_inputs.to(self.device)

                if isinstance(chat_inputs, torch.Tensor):
                    model_inputs = {
                        "input_ids": chat_inputs,
                        "attention_mask": torch.ones_like(chat_inputs, device=self.device),
                    }
                    prompt_length = chat_inputs.shape[-1]
                else:
                    model_inputs = {name: tensor.to(self.device) for name, tensor in chat_inputs.items()}
                    if "attention_mask" not in model_inputs:
                        model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"], device=self.device)
                    prompt_length = model_inputs["input_ids"].shape[-1]

                outputs = self.model.generate(
                    **model_inputs,
                    **generation_kwargs,
                )
                generated_tokens = outputs[0][prompt_length:]
                return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            prompt = render_plain_prompt(observation)
            encoded = self.tokenizer(prompt, return_tensors="pt")
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            outputs = self.model.generate(**encoded, **generation_kwargs)
            generated_tokens = outputs[0][encoded["input_ids"].shape[-1] :]
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


@lru_cache(maxsize=1)
def load_model_policy_from_env() -> HuggingFaceModelPolicy | None:
    base_model = os.getenv("LOCAL_BASE_MODEL")
    adapter_path = os.getenv("LOCAL_ADAPTER_PATH")
    if not base_model:
        return None
    max_new_tokens = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "256"))
    allow_fallback = os.getenv("LOCAL_ALLOW_HEURISTIC_FALLBACK", "false").lower() == "true"
    return HuggingFaceModelPolicy(
        base_model=base_model,
        adapter_path=adapter_path,
        max_new_tokens=max_new_tokens,
        allow_heuristic_fallback=allow_fallback,
    )
