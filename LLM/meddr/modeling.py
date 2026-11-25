from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from meddr.constants import LORA_RANK


def parse_exact_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        unfenced = stripped
        if unfenced.startswith("```"):
            lines = unfenced.splitlines()
            if lines and lines[0].strip().casefold() in {
                "```",
                "```json",
            }:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            unfenced = "\n".join(lines).strip()
        object_start = unfenced.find("{")
        if object_start < 0:
            raise ValueError(
                "The model output must contain one valid JSON object."
            )
        try:
            value, _ = json.JSONDecoder().raw_decode(
                unfenced[object_start:]
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The model output must contain one valid JSON object."
            ) from exc
    if not isinstance(value, dict):
        raise ValueError("The model output must be a JSON object.")
    return value


def sequence_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    output_device = shifted_logits.device
    shifted_labels = labels[:, 1:].to(output_device)
    shifted_mask = completion_mask[:, 1:].to(
        device=output_device,
        dtype=torch.float32,
    )
    selected_logits = shifted_logits.gather(
        -1, shifted_labels.unsqueeze(-1)
    ).squeeze(-1).float()
    normalizer_chunks: list[torch.Tensor] = []
    for logits_chunk in shifted_logits.split(32, dim=1):
        if torch.is_grad_enabled() and logits_chunk.requires_grad:
            normalizer = checkpoint(
                lambda value: torch.logsumexp(
                    value.float(),
                    dim=-1,
                ),
                logits_chunk,
                use_reentrant=False,
            )
        else:
            normalizer = torch.logsumexp(
                logits_chunk.float(),
                dim=-1,
            )
        normalizer_chunks.append(normalizer)
    log_normalizer = torch.cat(normalizer_chunks, dim=1)
    selected = selected_logits - log_normalizer
    return (selected * shifted_mask).sum(dim=-1)


def prompt_completion_batch(
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not prompts or len(prompts) != len(completions):
        raise ValueError("Prompts and completions must be non-empty and aligned.")
    rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
        rows.append(torch.tensor(prompt_ids + completion_ids, dtype=torch.long))
        masks.append(
            torch.tensor(
                [0] * len(prompt_ids) + [1] * len(completion_ids),
                dtype=torch.long,
            )
        )
    maximum = max(row.numel() for row in rows)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(rows), maximum), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), maximum), dtype=torch.long)
    completion_mask = torch.zeros((len(rows), maximum), dtype=torch.long)
    for index, row in enumerate(rows):
        length = row.numel()
        input_ids[index, :length] = row
        attention_mask[index, :length] = 1
        completion_mask[index, :length] = masks[index]
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "completion_mask": completion_mask.to(device),
    }


class LocalChatModel:
    def __init__(
        self,
        model_name: str,
        *,
        device_map: str | dict[str, Any] | None = "balanced",
        trainable: bool = False,
        adapter_path: str | None = None,
        torch_dtype: str = "float16",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_by_name: dict[str, str | torch.dtype] = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if torch_dtype not in dtype_by_name:
            raise ValueError(
                "torch_dtype must be auto, float16, bfloat16, or float32."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_by_name[torch_dtype],
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.config.use_cache = False
        self.trainable = trainable
        if adapter_path:
            self._attach_adapter(
                adapter_path,
                is_trainable=trainable,
            )
        elif trainable:
            self._attach_adapter(None, is_trainable=True)
        if trainable:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        else:
            self.model.requires_grad_(False)
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def _attach_adapter(
        self,
        adapter_path: str | None,
        *,
        is_trainable: bool,
    ) -> None:
        from peft import LoraConfig, PeftModel, get_peft_model

        if adapter_path:
            path = Path(adapter_path)
            if not (path / "adapter_config.json").is_file():
                raise ValueError(
                    "The adapter directory must contain adapter_config.json."
                )
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=is_trainable,
            )
            return
        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=2 * LORA_RANK,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )

    def format_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ]
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return f"{system.strip()}\n\n{user.strip()}\n\n"
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    @torch.no_grad()
    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        disable_adapter: bool = False,
    ) -> tuple[str, str, dict[str, Any]]:
        prompt = self.format_prompt(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = temperature > 0.0
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        if temperature > 0.0:
            generation_config.temperature = temperature
            generation_config.top_p = top_p
            generation_config.top_k = 0
        else:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
        adapter_context = (
            self.model.disable_adapter()
            if disable_adapter and hasattr(self.model, "disable_adapter")
            else nullcontext()
        )
        with adapter_context:
            output = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )
        completion = self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        return prompt, completion, parse_exact_json_object(completion)

    def logprobs(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> torch.Tensor:
        batch = prompt_completion_batch(
            self.tokenizer,
            prompts,
            completions,
            self.device,
        )
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return sequence_logprob(
            output.logits,
            batch["input_ids"],
            batch["completion_mask"],
        )

    @torch.no_grad()
    def score_logprobs(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> torch.Tensor:
        return self.logprobs(prompts, completions).detach()

    def save_adapter(self, output_dir: str | Path) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(target)
        self.tokenizer.save_pretrained(target)


class FrozenChatModelView:
    """Expose a PEFT model's frozen base without loading another checkpoint."""

    def __init__(self, source: LocalChatModel) -> None:
        self.source = source

    @property
    def device(self) -> torch.device:
        return self.source.device

    def format_prompt(self, system: str, user: str) -> str:
        """Delegate prompt rendering to the shared tokenizer."""
        return self.source.format_prompt(system, user)

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[str, str, dict[str, Any]]:
        return self.source.generate(
            system,
            user,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            disable_adapter=True,
        )


@dataclass(frozen=True)
class QuestionAction:
    prompt: str
    completion: str
    objective_id: str
    question: str
    old_logprob: float
    validation_attempts: int = 1
    validation_rejections: tuple[str, ...] = ()
    semantic_repairs: tuple[str, ...] = ()
