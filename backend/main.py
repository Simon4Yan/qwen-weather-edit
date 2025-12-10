import os
import dashscope
from dashscope import MultiModalConversation
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS for frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
API_KEY = os.getenv("DASHSCOPE_API_KEY")


@app.post("/api/edit")
async def edit_image(
    prompt: str = Form(...),
    file: UploadFile = File(...)
):
    if not API_KEY:
        return {"success": False, "error": "Missing DASHSCOPE_API_KEY"}

    # ---- FIX: always enforce bytes ----
    raw = await file.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    image_bytes = raw  # ensure bytes only

    # Correct Qwen Image Edit Format
    messages = [
        {
            "role": "user",
            "content": [
                { "image": image_bytes },   # <-- MUST be bytes
                { "text": prompt }
            ]
        }
    ]

    response = MultiModalConversation.call(
        api_key=API_KEY,
        model="qwen-image-edit",
        messages=messages,
        stream=False,
        n=1,
        watermark=False,
        prompt_extend=True
    )

    if response.status_code == 200:
        url = response.output.choices[0].message.content[0]["image"]
        return {"success": True, "image_url": url}

    return {
        "success": False,
        "status_code": response.status_code,
        "error_message": response.message
    }


@app.get("/health")
def health():
    return {"status": "ok"}