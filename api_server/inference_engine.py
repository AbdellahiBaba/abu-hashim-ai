import os
import logging
from typing import Optional, Dict, Any, AsyncGenerator

logger = logging.getLogger("qalam_ai.inference")

MODEL_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_finetune")
DEFAULT_MODEL_NAME = "CohereForAI/aya-23-8B"


class InferenceEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name: Optional[str] = None
        self.device: str = "cpu"
        self.is_loaded: bool = False
        self._generation_defaults = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
        }

    def load_model(self, model_path: Optional[str] = None) -> bool:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if model_path is None:
                finetuned_path = os.path.join(MODEL_BASE_DIR, "final_model")
                if os.path.exists(finetuned_path):
                    model_path = finetuned_path
                else:
                    model_path = DEFAULT_MODEL_NAME

            logger.info(f"Loading model from: {model_path}")

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left",
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if self.device == "cuda":
                try:
                    import bitsandbytes
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    load_kwargs["quantization_config"] = quantization_config
                    load_kwargs["device_map"] = "auto"
                except ImportError:
                    load_kwargs["torch_dtype"] = torch.float16
                    load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            self.model_name = model_path
            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True

        except ImportError as e:
            logger.warning(f"ML libraries not available: {e}. Running in demo mode.")
            self.model_name = "demo-mode"
            self.is_loaded = False
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        params = {**self._generation_defaults, **kwargs}

        if not self.is_loaded or self.model is None:
            return self._demo_generate(prompt, params)

        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            prompt_tokens = inputs["input_ids"].shape[1]

            with torch.no_grad():
                generation_kwargs = {
                    "max_new_tokens": params["max_new_tokens"],
                    "temperature": max(params["temperature"], 0.01),
                    "top_p": params["top_p"],
                    "top_k": params["top_k"],
                    "repetition_penalty": params["repetition_penalty"],
                    "do_sample": params["do_sample"],
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }

                if not params["do_sample"]:
                    generation_kwargs.pop("temperature", None)
                    generation_kwargs.pop("top_p", None)
                    generation_kwargs.pop("top_k", None)

                outputs = self.model.generate(**inputs, **generation_kwargs)

            new_tokens = outputs[0][prompt_tokens:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            return {
                "generated_text": generated_text.strip(),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": len(new_tokens),
                "finish_reason": "stop",
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "generated_text": f"Error during generation: {str(e)}",
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "finish_reason": "error",
            }

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        if not self.is_loaded or self.model is None:
            result = self._demo_generate(prompt, {**self._generation_defaults, **kwargs})
            for word in result["generated_text"].split():
                yield word + " "
            return

        try:
            import torch
            from transformers import TextIteratorStreamer
            import threading

            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            params = {**self._generation_defaults, **kwargs}

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": params["max_new_tokens"],
                "temperature": max(params["temperature"], 0.01),
                "top_p": params["top_p"],
                "top_k": params["top_k"],
                "repetition_penalty": params["repetition_penalty"],
                "do_sample": params["do_sample"],
                "streamer": streamer,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            for text in streamer:
                yield text

            thread.join()

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}"

    def _demo_generate(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        demo_responses = {
            "ar": "مرحباً! أنا نموذج قلم الذكي (QalamAI). النموذج يعمل حالياً في وضع العرض التوضيحي. يرجى تحميل النموذج الكامل للحصول على استجابات حقيقية.",
            "en": "Hello! I am the QalamAI model. Currently running in demo mode. Please load the full model for real responses.",
        }

        is_arabic = any("\u0600" <= c <= "\u06FF" for c in prompt)
        response = demo_responses["ar"] if is_arabic else demo_responses["en"]

        return {
            "generated_text": response,
            "prompt_tokens": len(prompt.split()),
            "generated_tokens": len(response.split()),
            "finish_reason": "demo",
        }

    def format_chat_prompt(self, messages: list) -> str:
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user") if isinstance(msg, dict) else msg.role
            content = msg.get("content", "") if isinstance(msg, dict) else msg.content

            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"

        formatted += "Assistant: "
        return formatted

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_loaded": self.is_loaded,
            "model_name": self.model_name,
            "device": self.device,
            "mode": "inference" if self.is_loaded else "demo",
        }


engine = InferenceEngine()
