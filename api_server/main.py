import os
import sys
import uuid
import json
import time
import hashlib
import logging
from datetime import datetime
from typing import Optional
from collections import deque

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from api_server.schemas import (
    GenerateRequest,
    GenerateResponse,
    ChatRequest,
    ChatResponse,
    ChatMessage,
    HealthResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from api_server.inference_engine import engine
from api_server.safety_filters import safety_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("qalam_ai.server")

SERVER_PORT = int(os.environ.get("SERVER_PORT", "5000"))
SAFETY_ENABLED = os.environ.get("SAFETY_ENABLED", "true").lower() == "true"

app = FastAPI(
    title="QalamAI - Abu Hashim AI Model",
    description="FastAPI inference server for the QalamAI Arabic language model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
FEEDBACK_DIR = os.path.join(PROJECT_ROOT, "learning_buffer")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR) if os.path.isdir(TEMPLATES_DIR) else None

SESSION_SECRET = os.environ.get("SESSION_SECRET", "")
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
SESSION_MAX_AGE = 86400

if not SESSION_SECRET:
    logger.warning("SESSION_SECRET not set — generating a random one. Set it for persistent sessions.")
    SESSION_SECRET = uuid.uuid4().hex + uuid.uuid4().hex

if not ADMIN_USERNAME or not ADMIN_PASSWORD:
    logger.warning("ADMIN_USERNAME/ADMIN_PASSWORD not set — dashboard login will be disabled.")

serializer = URLSafeTimedSerializer(SESSION_SECRET)


def _create_session_token(username: str) -> str:
    return serializer.dumps({"user": username})


def _verify_session(token: str) -> Optional[str]:
    try:
        data = serializer.loads(token, max_age=SESSION_MAX_AGE)
        return data.get("user")
    except (BadSignature, SignatureExpired):
        return None


def _is_authenticated(request: Request) -> bool:
    token = request.cookies.get("qalam_session")
    if not token:
        return False
    return _verify_session(token) is not None


request_log = deque(maxlen=1000)
latency_log = deque(maxlen=100)


def record_request(req_type, latency_ms, status="success", details=""):
    entry = {
        "time": datetime.utcnow().isoformat(),
        "type": req_type,
        "status": status,
        "latency_ms": round(latency_ms, 1),
        "details": details,
    }
    request_log.append(entry)
    latency_log.append(latency_ms)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting QalamAI Inference Server...")
    model_path = os.environ.get("QALAM_MODEL_PATH", None)
    model_name = os.environ.get("MODEL_NAME", None)
    engine.load_model(model_path)
    status = engine.get_status()
    logger.info(f"Engine status: {status}")
    logger.info(f"Safety filters: {'enabled' if SAFETY_ENABLED else 'disabled'}")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _is_authenticated(request):
        return RedirectResponse(url="/", status_code=302)
    if templates:
        return templates.TemplateResponse("login.html", {"request": request, "error": None})
    return HTMLResponse("<html><body><h1>Login</h1><p>Templates not configured.</p></body></html>")


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    if ADMIN_USERNAME and ADMIN_PASSWORD and username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        token = _create_session_token(username)
        response = RedirectResponse(url="/", status_code=302)
        is_https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(
            key="qalam_session",
            value=token,
            max_age=SESSION_MAX_AGE,
            httponly=True,
            secure=is_https,
            samesite="lax",
            path="/",
        )
        return response
    if templates:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password"})
    return HTMLResponse("<html><body><h1>Login Failed</h1></body></html>", status_code=401)


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("qalam_session")
    return response


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    auth_configured = bool(ADMIN_USERNAME and ADMIN_PASSWORD)
    if auth_configured and not _is_authenticated(request):
        return RedirectResponse(url="/login", status_code=302)
    if templates and os.path.exists(os.path.join(TEMPLATES_DIR, "dashboard.html")):
        return templates.TemplateResponse("dashboard.html", {"request": request})
    return HTMLResponse(
        content="<html><body><h1>QalamAI Server</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p></body></html>"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    status = engine.get_status()
    return HealthResponse(
        status="healthy",
        model_loaded=status["is_loaded"],
        model_name=status.get("model_name"),
    )


@app.get("/api/health", response_model=HealthResponse)
async def api_health_check():
    status = engine.get_status()
    return HealthResponse(
        status="healthy",
        model_loaded=status["is_loaded"],
        model_name=status.get("model_name"),
    )


@app.post("/api/generate", response_model=GenerateResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def generate(request: GenerateRequest):
    start = time.time()

    if SAFETY_ENABLED:
        is_safe, reason = safety_filter.check_input(request.prompt)
        if not is_safe:
            record_request("generate", (time.time() - start) * 1000, "blocked", reason)
            raise HTTPException(status_code=400, detail=reason)

    if request.stream:
        return StreamingResponse(
            _stream_generate(request),
            media_type="text/event-stream",
        )

    result = engine.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    )

    filtered_text = safety_filter.filter_output(result["generated_text"]) if SAFETY_ENABLED else result["generated_text"]
    elapsed = (time.time() - start) * 1000
    record_request("generate", elapsed, "success", f"{result['generated_tokens']} tokens")

    return GenerateResponse(
        generated_text=filtered_text,
        prompt_tokens=result["prompt_tokens"],
        generated_tokens=result["generated_tokens"],
        finish_reason=result["finish_reason"],
    )


async def _stream_generate(request: GenerateRequest):
    async for chunk in engine.generate_stream(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    ):
        filtered = safety_filter.filter_output(chunk) if SAFETY_ENABLED else chunk
        yield f"data: {json.dumps({'text': filtered})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/chat", response_model=ChatResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def chat(request: ChatRequest):
    start = time.time()
    last_message = request.messages[-1].content

    if SAFETY_ENABLED:
        is_safe, reason = safety_filter.check_input(last_message)
        if not is_safe:
            record_request("chat", (time.time() - start) * 1000, "blocked", reason)
            raise HTTPException(status_code=400, detail=reason)

    prompt = engine.format_chat_prompt([m.model_dump() for m in request.messages])

    if request.stream:
        return StreamingResponse(
            _stream_chat(prompt, request),
            media_type="text/event-stream",
        )

    result = engine.generate(
        prompt=prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    )

    filtered_text = safety_filter.filter_output(result["generated_text"]) if SAFETY_ENABLED else result["generated_text"]
    elapsed = (time.time() - start) * 1000
    record_request("chat", elapsed, "success", f"{result['generated_tokens']} tokens")

    return ChatResponse(
        message=ChatMessage(role="assistant", content=filtered_text),
        prompt_tokens=result["prompt_tokens"],
        generated_tokens=result["generated_tokens"],
        finish_reason=result["finish_reason"],
    )


async def _stream_chat(prompt: str, request: ChatRequest):
    async for chunk in engine.generate_stream(
        prompt=prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    ):
        filtered = safety_filter.filter_output(chunk) if SAFETY_ENABLED else chunk
        yield f"data: {json.dumps({'text': filtered})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    feedback_id = str(uuid.uuid4())

    feedback_data = {
        "id": feedback_id,
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": request.prompt,
        "response": request.response,
        "rating": request.rating,
        "comment": request.comment,
    }

    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    feedback_file = os.path.join(FEEDBACK_DIR, f"feedback_{feedback_id}.json")
    with open(feedback_file, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Feedback recorded: {feedback_id} (rating: {request.rating})")
    record_request("feedback", 0, "success", f"rating={request.rating}")

    return FeedbackResponse(status="recorded", feedback_id=feedback_id)


@app.get("/api/status")
async def model_status():
    return engine.get_status()


@app.get("/api/stats")
async def get_stats():
    buffer_count = 0
    if os.path.isdir(FEEDBACK_DIR):
        buffer_count = len([f for f in os.listdir(FEEDBACK_DIR) if f.endswith(".json")])

    status = engine.get_status()
    avg_lat = round(sum(latency_log) / len(latency_log), 1) if latency_log else None

    return {
        "total_requests": len(request_log),
        "avg_latency": avg_lat,
        "active_models": 1 if status["is_loaded"] else 0,
        "buffer_size": buffer_count,
        "recent_activity": list(request_log)[-10:],
    }


@app.post("/api/evaluate")
async def run_evaluation():
    start = time.time()
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from evaluation.evaluate import EvaluationRunner

        runner = EvaluationRunner()
        results = runner.run_quick_evaluation(engine.generate)
        elapsed = (time.time() - start) * 1000
        record_request("evaluation", elapsed, "success", f"{len(results.get('results', []))} tests")
        return {"status": "completed", "results": results}
    except ImportError:
        record_request("evaluation", (time.time() - start) * 1000, "error", "evaluation module not available")
        return {"status": "error", "message": "Evaluation module not available. Ensure evaluation package is properly installed."}
    except Exception as e:
        record_request("evaluation", (time.time() - start) * 1000, "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/update-model")
async def update_model():
    start = time.time()
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from training_scripts.update_model import update_model as run_update

        result = run_update(dry_run=True)
        elapsed = (time.time() - start) * 1000
        record_request("update_model", elapsed, "success", "dry run completed")
        return {"status": "completed", "result": result}
    except ImportError:
        record_request("update_model", (time.time() - start) * 1000, "error", "update module not available")
        return {"status": "error", "message": "Update module not available. Ensure training_scripts package is properly installed."}
    except Exception as e:
        record_request("update_model", (time.time() - start) * 1000, "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process-buffer")
async def process_buffer():
    start = time.time()
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from training_scripts.self_learning import run_self_learning_cycle

        result = run_self_learning_cycle()
        elapsed = (time.time() - start) * 1000
        record_request("process_buffer", elapsed, "success", f"merged={result.get('merged_count', 0)}")
        return {"status": "completed", "result": result}
    except ImportError as e:
        record_request("process_buffer", (time.time() - start) * 1000, "error", str(e))
        return {"status": "error", "message": "Self-learning module not available."}
    except Exception as e:
        record_request("process_buffer", (time.time() - start) * 1000, "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))


QALAM_EXPORTS_DIR = os.path.join(PROJECT_ROOT, "dataset_raw", "qalam_exports")
QALAM_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset_processed", "qalam_processed")


MAX_UPLOAD_SIZE = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {".json", ".csv", ".jsonl"}


def _sanitize_filename(filename: str) -> str:
    import re as _re
    name = os.path.basename(filename)
    name = _re.sub(r"[^\w\-.]", "_", name)
    if not name or name.startswith("."):
        name = f"upload_{uuid.uuid4().hex[:8]}{os.path.splitext(filename)[1]}"
    return name


@app.post("/api/qalam-import")
async def qalam_import(file: UploadFile = File(...)):
    start = time.time()
    try:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

        sys.path.insert(0, PROJECT_ROOT)
        from qalam_bridge.importer import QalamImporter

        os.makedirs(QALAM_EXPORTS_DIR, exist_ok=True)
        safe_name = _sanitize_filename(file.filename)
        temp_path = os.path.join(QALAM_EXPORTS_DIR, safe_name)

        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE // (1024*1024)}MB")

        with open(temp_path, "wb") as f:
            f.write(content)

        importer = QalamImporter()
        records = importer.import_file(temp_path)
        import_result = {
            "accepted": importer.stats.get("records_accepted", 0),
            "rejected": importer.stats.get("records_rejected", 0),
            "duplicates": importer.stats.get("duplicates_skipped", 0),
            "total_read": importer.stats.get("records_read", 0),
        }

        elapsed = (time.time() - start) * 1000
        record_request("qalam_import", elapsed, "success", f"imported={import_result['accepted']}")
        return {"status": "completed", "result": import_result}
    except HTTPException:
        raise
    except ImportError as e:
        record_request("qalam_import", (time.time() - start) * 1000, "error", str(e))
        return {"status": "error", "message": f"QalamAI bridge module not available: {e}"}
    except Exception as e:
        record_request("qalam_import", (time.time() - start) * 1000, "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/qalam-stats")
async def qalam_stats():
    try:
        sys.path.insert(0, PROJECT_ROOT)

        raw_count = 0
        if os.path.isdir(QALAM_EXPORTS_DIR):
            raw_count = len([f for f in os.listdir(QALAM_EXPORTS_DIR) if not f.startswith(".")])

        processed_count = 0
        categories = {}
        quality_sum = 0.0
        scored_count = 0

        jsonl_sources = []
        if os.path.isdir(QALAM_PROCESSED_DIR):
            for fname in os.listdir(QALAM_PROCESSED_DIR):
                if fname.endswith(".jsonl"):
                    jsonl_sources.append(os.path.join(QALAM_PROCESSED_DIR, fname))

        main_dataset = os.path.join(PROJECT_ROOT, "dataset_processed", "qalam_training_data.jsonl")
        if os.path.isfile(main_dataset):
            jsonl_sources.append(main_dataset)

        for fpath in jsonl_sources:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        processed_count += 1
                        try:
                            record = json.loads(line)
                            cat = record.get("category", "unknown")
                            categories[cat] = categories.get(cat, 0) + 1
                            q = record.get("quality")
                            if q is not None:
                                quality_sum += float(q)
                                scored_count += 1
                        except (json.JSONDecodeError, ValueError):
                            pass

        train_file = os.path.join(PROJECT_ROOT, "dataset_processed", "train.jsonl")
        eval_file = os.path.join(PROJECT_ROOT, "dataset_processed", "eval.jsonl")
        train_count = 0
        eval_count = 0
        if os.path.isfile(train_file):
            with open(train_file, "r") as f:
                train_count = sum(1 for line in f if line.strip())
        if os.path.isfile(eval_file):
            with open(eval_file, "r") as f:
                eval_count = sum(1 for line in f if line.strip())

        return {
            "raw_exports": raw_count,
            "processed_records": processed_count,
            "categories": categories,
            "avg_quality": round(quality_sum / scored_count, 3) if scored_count > 0 else None,
            "train_records": train_count,
            "eval_records": eval_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")

webhook_counters = {
    "total_received": 0,
    "accepted": 0,
    "rejected": 0,
    "duplicates": 0,
    "errors": 0,
    "last_received": None,
}

_webhook_importer = None


def _get_webhook_importer():
    global _webhook_importer
    if _webhook_importer is None:
        sys.path.insert(0, PROJECT_ROOT)
        from qalam_bridge.importer import QalamImporter
        _webhook_importer = QalamImporter()
    return _webhook_importer


@app.post("/api/qalam-webhook")
async def qalam_webhook(request: Request):
    start = time.time()

    secret = request.headers.get("X-Webhook-Secret", "")
    if not WEBHOOK_SECRET or secret != WEBHOOK_SECRET:
        webhook_counters["total_received"] += 1
        webhook_counters["errors"] += 1
        record_request("webhook", (time.time() - start) * 1000, "error", "auth_failed")
        raise HTTPException(status_code=401, detail="Invalid or missing webhook secret")

    try:
        payload = await request.json()
    except Exception:
        webhook_counters["total_received"] += 1
        webhook_counters["errors"] += 1
        record_request("webhook", (time.time() - start) * 1000, "error", "bad_json")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if not isinstance(payload, dict):
        webhook_counters["total_received"] += 1
        webhook_counters["errors"] += 1
        record_request("webhook", (time.time() - start) * 1000, "error", "not_object")
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")

    webhook_counters["total_received"] += 1
    webhook_counters["last_received"] = datetime.utcnow().isoformat() + "Z"

    try:
        importer = _get_webhook_importer()
        result = importer.import_single_record(payload)

        status = result.get("status", "rejected")
        if status == "accepted":
            webhook_counters["accepted"] += 1
        elif status == "duplicate":
            webhook_counters["duplicates"] += 1
        else:
            webhook_counters["rejected"] += 1

        elapsed = (time.time() - start) * 1000
        record_request("webhook", elapsed, "success", f"status={status}")
        return {"status": status, "quality": result.get("quality"), "hash": result.get("hash")}
    except ImportError as e:
        webhook_counters["errors"] += 1
        record_request("webhook", (time.time() - start) * 1000, "error", str(e))
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Bridge module not available: {e}"})
    except Exception as e:
        webhook_counters["errors"] += 1
        record_request("webhook", (time.time() - start) * 1000, "error", str(e))
        logger.error("Webhook processing error: %s", e)
        return JSONResponse(status_code=500, content={"status": "error", "message": "Internal processing error"})


@app.get("/api/qalam-webhook-stats")
async def qalam_webhook_stats():
    return dict(webhook_counters)


@app.post("/api/qalam-build-dataset")
async def qalam_build_dataset():
    start = time.time()
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from qalam_bridge.dataset_builder import build_dataset

        result = build_dataset()
        elapsed = (time.time() - start) * 1000
        record_request("qalam_build", elapsed, "success", f"train={result.get('stats', {}).get('train_records', 0)}")
        return {"status": "completed", "result": result}
    except ImportError as e:
        record_request("qalam_build", (time.time() - start) * 1000, "error", str(e))
        return {"status": "error", "message": f"Dataset builder not available: {e}"}
    except Exception as e:
        record_request("qalam_build", (time.time() - start) * 1000, "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))
