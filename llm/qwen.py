import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenLLM:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        max_input_tokens: int = 3072,
        temperature: float = 0.0,
    ):
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str:
        text = self._format_prompt(prompt)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=False,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}


        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        input_length = inputs["input_ids"].shape[1]
        generated = outputs[0, input_length:]

        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        if not prompts:
            return []

        texts = [self._format_prompt(prompt) for prompt in prompts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            pad_to_multiple_of=8,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        input_length = inputs["input_ids"].shape[1]

        results = []
        for output in outputs:
            generated = output[input_length:]
            results.append(
                self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            )

        return results

    @staticmethod
    def extract_json(text: str):
        text = text.strip()
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*", "", text)

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None