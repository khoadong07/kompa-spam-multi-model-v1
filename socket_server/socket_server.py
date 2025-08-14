import socketio
import uvicorn
from fastapi import FastAPI
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError, retry_if_exception_type

# Tạo Socket.IO server kết hợp với FastAPI
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI()
asgi_app = socketio.ASGIApp(sio, app)

INFER_URL = "http://0.0.0.0:8989/predict"

OUTPUT_FIELDS = [
    "id", "topic", "topic_id", "title", "content", "description",
    "sentiment", "site_name", "site_id", "type", "category", "spam", "lang", "label", "label_id"
]

# Global aiohttp session
aiohttp_session: aiohttp.ClientSession = None

@app.on_event("startup")
async def startup():
    global aiohttp_session
    aiohttp_session = aiohttp.ClientSession()

@app.on_event("shutdown")
async def shutdown():
    await aiohttp_session.close()

# Retry config: max 3 lần, mỗi lần cách 2s
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def call_inference(text: str, category: str):
    async with aiohttp_session.post(INFER_URL, json={"text": text, "category": category}, timeout=10) as resp:
        if resp.status != 200:
            raise aiohttp.ClientError(f"HTTP {resp.status}")
        return await resp.json()

@sio.event
async def connect(sid, environ):
    print(f"🔌 Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")

@sio.on("predict")
async def handle_predict(sid, payload):
    print(f"📥 Received 'predict' from {sid}")

    category = payload.get("category", "")
    data_batch = payload.get("data", [])

    tasks = []
    for item in data_batch:
        title = item.get("title", "") or ""
        description = item.get("description", "") or ""
        content = item.get("content", "") or ""
        text = f"{title}\n{description}\n{content}".strip()

        task = asyncio.create_task(process_item(item, text, category))
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    print(f"📤 Emitting 'result' to {sid}")
    await sio.emit("result", {
        "category": category,
        "results": results
    }, to=sid)

async def process_item(item, text, category):
    try:
        prediction = await call_inference(text, category)
    except RetryError as e:
        print(f"⚠️ Retry failed for item {item.get('id')}: {e}")
        prediction = {"spam": None, "lang": None}

    merged = {
        **item,
        "spam": prediction.get("spam", None),
        "lang": prediction.get("lang", None),
        "category": category
    }

    # call predict_ads if spam is True
    if merged.get("spam") is True:
        from ads_predict import predict_ads
        try:
            print(merged['content'])
            is_ads = predict_ads(merged['content'])
            print(is_ads)
            if is_ads:
                merged["label"] = 'Rao vặt'
                merged["label_id"] = '68898a3c16a3634d83338269'
                merged["sentiment"] = 'Neutral'
            else:
                merged["label"] = None
                merged["label_id"] = None
                merged["sentiment"] = None
        except Exception as e:
            print(f"⚠️ Error in predict_ads for item {item.get('id')}: {e}")
            merged["is_ads"] = None
    # print(f"✅ Processed item: {merged}")
    
    return {k: merged.get(k, None) for k in OUTPUT_FIELDS}

if __name__ == "__main__":
    uvicorn.run(asgi_app, host="0.0.0.0", port=5001, workers=1)
