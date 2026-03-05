# Integration Guide — Abu Hashim / QalamAI

## Overview

This guide explains how to integrate Abu Hashim into your applications, whether through the REST API, direct Python usage, or embedding the model in your own infrastructure.

## Option 1: REST API Integration

The simplest way to integrate Abu Hashim is through its FastAPI inference server.

### Starting the Server

```bash
python main.py
```

The server runs on `http://localhost:5000`.

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:5000"

def generate_text(prompt, temperature=0.7, max_tokens=512):
    response = requests.post(f"{BASE_URL}/generate", json={
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_tokens,
        "do_sample": True,
    })
    return response.json()["generated_text"]

def chat(messages, temperature=0.7):
    response = requests.post(f"{BASE_URL}/chat", json={
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "temperature": temperature,
        "do_sample": True,
    })
    return response.json()["message"]["content"]

result = generate_text("ما هو الذكاء الاصطناعي؟")
print(result)

reply = chat([
    {"role": "user", "content": "كيف حالك؟"}
])
print(reply)
```

### JavaScript/Node.js Client Example

```javascript
const BASE_URL = "http://localhost:5000";

async function generateText(prompt, options = {}) {
  const response = await fetch(`${BASE_URL}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      temperature: options.temperature || 0.7,
      max_new_tokens: options.maxTokens || 512,
      do_sample: true,
    }),
  });
  const data = await response.json();
  return data.generated_text;
}

async function chat(messages, options = {}) {
  const response = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      temperature: options.temperature || 0.7,
      do_sample: true,
    }),
  });
  const data = await response.json();
  return data.message.content;
}
```

### cURL Examples

Text generation:
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ما هو الذكاء الاصطناعي؟", "max_new_tokens": 256}'
```

Chat completion:
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "كيف حالك؟"}]}'
```

Health check:
```bash
curl http://localhost:5000/health
```

## Option 2: Direct Python Usage

For tighter integration, import the inference engine directly:

```python
from model_base.config import get_default_config
from api_server.inference_engine import InferenceEngine

config = get_default_config()
engine = InferenceEngine(config)
engine.load_model()

result = engine.generate("ما هو الذكاء الاصطناعي؟", max_new_tokens=256)
print(result)
```

## Option 3: Streaming Integration

For real-time applications, use the streaming endpoint:

```python
import requests

def stream_generate(prompt):
    response = requests.post(
        "http://localhost:5000/generate",
        json={"prompt": prompt, "stream": True},
        stream=True,
    )
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        print(chunk, end="", flush=True)
```

## Submitting User Feedback

Collect user feedback to improve the model over time:

```python
def submit_feedback(prompt, response, rating, comment=None):
    requests.post("http://localhost:5000/feedback", json={
        "prompt": prompt,
        "response": response,
        "rating": rating,
        "comment": comment,
    })
```

## Error Handling

Always handle potential errors from the API:

```python
import requests

def safe_generate(prompt):
    try:
        response = requests.post("http://localhost:5000/generate", json={
            "prompt": prompt,
            "max_new_tokens": 256,
        }, timeout=60)
        response.raise_for_status()
        return response.json()["generated_text"]
    except requests.exceptions.ConnectionError:
        return "Server is not available"
    except requests.exceptions.Timeout:
        return "Request timed out"
    except requests.exceptions.HTTPError as e:
        error_data = e.response.json()
        return f"Error: {error_data.get('error', 'Unknown error')}"
```

## Environment Variables

| Variable               | Description                          | Default     |
|------------------------|--------------------------------------|-------------|
| `MODEL_NAME`           | HuggingFace model identifier         | aya-23-8b   |
| `MODEL_CACHE_DIR`      | Directory for model weights          | model_base/weights |
| `SERVER_PORT`          | Inference server port                | 5000        |
| `SAFETY_ENABLED`       | Enable safety filters                | true        |
| `SAFETY_STRICT_MODE`   | Enable strict safety mode            | false       |

## Rate Limiting

For production deployments, consider adding rate limiting via a reverse proxy (e.g., Nginx) or middleware. The server does not include built-in rate limiting.

## CORS

The FastAPI server includes CORS middleware. Configure allowed origins for your deployment environment.
