import argparse
import os
import sys

from huggingface_hub import snapshot_download

from .config import ModelConfig, SUPPORTED_MODELS, get_default_config


def download_model(config: ModelConfig | None = None, model_key: str | None = None) -> str:
    if config is None:
        config = get_default_config()

    model_name = config.model_name
    if model_key and model_key in SUPPORTED_MODELS:
        model_name = SUPPORTED_MODELS[model_key]["name"]

    cache_dir = config.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading model: {model_name}")
    print(f"Cache directory: {cache_dir}")

    local_dir = snapshot_download(
        repo_id=model_name,
        revision=config.model_revision,
        cache_dir=cache_dir,
        resume_download=True,
    )

    print(f"Model downloaded to: {local_dir}")
    return local_dir


def download_tokenizer(config: ModelConfig | None = None) -> str:
    if config is None:
        config = get_default_config()

    tokenizer_name = config.tokenizer_name or config.model_name
    cache_dir = config.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading tokenizer: {tokenizer_name}")

    local_dir = snapshot_download(
        repo_id=tokenizer_name,
        revision=config.model_revision,
        cache_dir=cache_dir,
        resume_download=True,
        allow_patterns=["tokenizer*", "special_tokens*", "vocab*", "*.json"],
    )

    print(f"Tokenizer downloaded to: {local_dir}")
    return local_dir


def list_available_models():
    print("Available models for Abu Hashim / QalamAI:")
    print("-" * 60)
    for key, info in SUPPORTED_MODELS.items():
        rec = " [RECOMMENDED]" if info.get("recommended") else ""
        print(f"  {key}: {info['description']} ({info['parameters']}){rec}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Download base model for Abu Hashim / QalamAI")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model key to download (default: aya-23-8b)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory for model weights",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    args = parser.parse_args()

    if args.list:
        list_available_models()
        sys.exit(0)

    config = get_default_config()
    if args.cache_dir:
        config.cache_dir = args.cache_dir

    download_model(config=config, model_key=args.model)


if __name__ == "__main__":
    main()
