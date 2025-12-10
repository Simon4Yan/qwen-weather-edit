import os
import dashscope
from dashscope import MultiModalConversation
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------------------------------------
# FastAPI application setup
# ----------------------------------------------------------
app = FastAPI(title="Qwen Weather Edit Backend")

# Allow frontend (GitHub Pages / local) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# DashScope international endpoint (Singapore region)
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

# Read API key from Render's environment variables
API_KEY = os.getenv("DASHSCOPE_API_KEY")


# ----------------------------------------------------------
# POST /api/edit
# Accepts:  (1) image file  (2) text prompt
# Returns:  URL of edited image
# ----------------------------------------------------------
@app.post("/api/edit")
async def edit_image(
    prompt: str = Form(...),
    file: UploadFile = File(...)
):
    # Check for API key
    if not API_KEY:
        return {
            "success": False,
            "error": "DASHSCOPE_API_KEY is missing from environment variables."
        }

    # Read uploaded image bytes
    image_bytes = await file.read()

    # IMPORTANT:
    # qwen-image-edit expects the image content directly as bytes,
    # NOT a dict with {"type": ..., "data": ...}
    messages = [
        {
            "role": "user",
            "content": [
                { "image": image_bytes },   # <-- Correct format
                { "text": prompt }
            ]
        }
    ]

    # Call DashScope API
    response = MultiModalConversation.call(
        api_key=API_KEY,
        model="qwen-image-edit",
        messages=messages,
        stream=False,
        n=1,
        watermark=False,
        prompt_extend=True
    )

    # Success
    if response.status_code == 200:
        output_image_url = response.output.choices[0].message.content[0]["image"]
        return {
            "success": True,
            "image_url": output_image_url
        }

    # Error response
    return {
        "success": False,
        "status_code": response.status_code,
        "error_message": response.message
    }


# ----------------------------------------------------------
# GET /health
# Used by Render to verify the service is alive
# ----------------------------------------------------------
@app.get("/health")
def health():
    return { "status": "ok" }