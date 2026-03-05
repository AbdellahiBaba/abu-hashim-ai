# Inference Server — Abu Hashim / QalamAI

## Overview

Abu Hashim includes a FastAPI-based inference server for serving the fine-tuned model. The server supports text generation, chat completions, streaming responses, safety filtering, and user feedback collection.

## Starting the Server

```bash
python main.py
```

The server starts on port **5000** by default.

## API Endpoints

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "CohereForAI/aya-23-8B"
}
```

### Text Generation

```
POST /generate
```

Request body:
```json
{
  "prompt": "ما هو الذكاء الاصطناعي؟",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "do_sample": true,
  "stream": false,
  "stop_sequences": null
}
```

Response:
```json
{
  "generated_text": "الذكاء الاصطناعي هو فرع من علوم الحاسوب...",
  "prompt_tokens": 15,
  "generated_tokens": 128,
  "finish_reason": "stop"
}
```

### Chat Completion

```
POST /chat
```

Request body:
```json
{
  "messages": [
    {"role": "system", "content": "أنت أبو هاشم"},
    {"role": "user", "content": "كيف حالك؟"}
  ],
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "do_sample": true,
  "stream": false
}
```

Response:
```json
{
  "message": {
    "role": "assistant",
    "content": "أنا بخير، شكراً لسؤالك..."
  },
  "prompt_tokens": 20,
  "generated_tokens": 45,
  "finish_reason": "stop"
}
```

### User Feedback

```
POST /feedback
```

Request body:
```json
{
  "prompt": "ما هو الذكاء الاصطناعي؟",
  "response": "الذكاء الاصطناعي هو...",
  "rating": 5,
  "comment": "إجابة ممتازة"
}
```

Response:
```json
{
  "status": "recorded",
  "feedback_id": "uuid-string"
}
```

## Generation Modes

| Mode          | Key Parameter     | Description                    |
|---------------|-------------------|--------------------------------|
| Greedy        | `do_sample=false` | Deterministic, picks top token |
| Sampling      | `do_sample=true`  | Stochastic, uses temperature   |
| Beam Search   | N/A               | Uses beam search decoding      |

## Generation Parameters

| Parameter            | Type    | Range     | Default | Description                    |
|----------------------|---------|-----------|---------|--------------------------------|
| `max_new_tokens`     | int     | 1–4096    | 512     | Maximum tokens to generate     |
| `temperature`        | float   | 0.0–2.0   | 0.7     | Sampling temperature           |
| `top_p`              | float   | 0.0–1.0   | 0.9     | Nucleus sampling threshold     |
| `top_k`              | int     | 0–200     | 50      | Top-k sampling                 |
| `repetition_penalty` | float   | 1.0–2.0   | 1.1     | Penalty for repeated tokens    |
| `stream`             | bool    | —         | false   | Enable streaming response      |

## Safety Filters

The server includes built-in safety filters that check both input and output:

- **Input filtering**: Blocks requests containing harmful content patterns (weapons, hacking, self-harm)
- **Output filtering**: Scans generated text for unsafe content before returning
- **Strict mode**: Optionally enables profanity filtering and sensitive topic blocking
- **Bilingual support**: Filters work for both Arabic and English content

When content is blocked, the server returns a safety message in Arabic and English.

## Streaming

Set `"stream": true` in the request to receive Server-Sent Events (SSE) with token-by-token output.

## Admin Dashboard

The server serves an admin dashboard at the root URL (`/`) with:

- Model status and health information
- API usage statistics
- Interactive testing interface
- System monitoring

## Error Handling

All errors return a structured response:

```json
{
  "error": "Error type description",
  "detail": "Detailed error information"
}
```

Common HTTP status codes:

| Code | Meaning                    |
|------|----------------------------|
| 200  | Successful generation      |
| 400  | Invalid request parameters |
| 422  | Validation error           |
| 500  | Internal server error      |
| 503  | Model not loaded           |
