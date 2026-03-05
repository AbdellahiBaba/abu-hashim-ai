import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "CohereForAI/aya-23-8B"
    model_revision: str = "main"
    cache_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_base", "weights")

    tokenizer_name: Optional[str] = None
    trust_remote_code: bool = True

    quantization: str = "4bit"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    max_seq_length: int = 2048
    dtype: str = "bfloat16"

    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    generation_max_new_tokens: int = 512
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    generation_top_k: int = 50
    generation_repetition_penalty: float = 1.1

    system_prompt: str = (
        "أنت أبو هاشم، مساعد ذكاء اصطناعي عربي متقدم. "
        "أجب بدقة ووضوح باللغة العربية."
    )


def get_default_config() -> ModelConfig:
    return ModelConfig()


SUPPORTED_MODELS = {
    "aya-23-8b": {
        "name": "CohereForAI/aya-23-8B",
        "description": "Aya 23 8B - Arabic-capable multilingual model by Cohere",
        "parameters": "8B",
        "recommended": True,
    },
    "aya-23-35b": {
        "name": "CohereForAI/aya-23-35B",
        "description": "Aya 23 35B - Larger Arabic-capable multilingual model",
        "parameters": "35B",
        "recommended": False,
    },
    "jais-13b": {
        "name": "inception-mbzuai/jais-13b-chat",
        "description": "JAIS 13B Chat - Arabic-English bilingual model",
        "parameters": "13B",
        "recommended": False,
    },
    "qwen2-7b": {
        "name": "Qwen/Qwen2-7B-Instruct",
        "description": "Qwen2 7B - Multilingual model with Arabic support",
        "parameters": "7B",
        "recommended": False,
    },
}
