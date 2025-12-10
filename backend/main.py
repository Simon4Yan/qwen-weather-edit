import os
import base64
import dashscope
from dashscope import MultiModalConversation
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# FastAPI App Setup
# --------------------------------------------------
app = FastAPI(title="Qwen Weather Edit API")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow all origins (GitHub Pages, local, etc.)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use DashScope international endpoint (Singapore)
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

# API KEY from environment variable
API_KEY = os.getenv("DASHSCOPE_API_KEY")


# --------------------------------------------------
# POST /api/edit
# Image editing endpoint
# --------------------------------------------------
@app.post("/api/edit")
async def edit_image(
    prompt: str = Form(...),
    file: UploadFile = File(...)
):
    """Receive image + prompt, call Qwen Image Edit, return edited image URL."""

    if not API_KEY:
        return {"success": False, "error": "Missing DASHSCOPE_API_KEY"}

    # -------------------------
    # Read uploaded file safely
    # -------------------------
    raw = await file.read()

    # Render may convert to str → convert back to bytes
    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    # Convert to base64 string with MIME header
    b64 = base64.b64encode(raw).decode("utf-8")
    image_data_url = f"data:{file.content_type};base64,{b64}"

    # -------------------------
    # Qwen MultiModal API format
    # -------------------------
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_data_url},   # MUST be base64 dataURL
                {"text": prompt}
            ]
        }
    ]

    # -------------------------
    # Call DashScope API
    # -------------------------
    response = MultiModalConversation.call(
        api_key=API_KEY,
        model="qwen-image-edit",
        messages=messages,
        stream=False,
        n=1,                # only 1 edited result
        watermark=False,
        prompt_extend=True
    )

    # -------------------------
    # Success Case
    # -------------------------
    if response.status_code == 200:
        try:
            image_url = response.output.choices[0].message.content[0]["image"]
        except:
            return {
                "success": False,
                "error_message": "Qwen returned invalid image data."
            }

        return {
            "success": True,
            "image_url": image_url
        }

    # -------------------------
    # Error Case
    # -------------------------
    return {
        "success": False,
        "status_code": response.status_code,
        "error_message": response.message
    }


# --------------------------------------------------
# GET /health
# Render health check
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}