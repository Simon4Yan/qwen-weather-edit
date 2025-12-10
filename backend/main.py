import os
import dashscope
from dashscope import MultiModalConversation
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------------------------------------
# FastAPI application setup
# ----------------------------------------------------------
app = FastAPI(title="Qwen Image Edit Backend")

# Allow cross-origin requests so your frontend website
# (e.g., GitHub Pages) can call this backend API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow all origins (simple for deployment)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use DashScope Singapore region endpoint
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

# Read the API key from environment variable
API_KEY = os.getenv("DASHSCOPE_API_KEY")


# ----------------------------------------------------------
# Main API endpoint: /api/edit
# - Accepts an uploaded image
# - Accepts a text editing prompt
# - Sends both to Qwen Image Edit model
# - Returns a generated image URL
# ----------------------------------------------------------
@app.post("/api/edit")
async def edit_image(
    prompt: str = Form(...),         # Editing instruction (text)
    file: UploadFile = File(...),    # Image uploaded by user
):
    # Ensure API key exists
    if not API_KEY:
        return {
            "success": False,
            "error": "DASHSCOPE_API_KEY is missing. Please set the environment variable."
        }

    # Read the raw image bytes directly
    # This avoids the size limits of data-URI and base64 encoding
    image_bytes = await file.read()

    # Build DashScope multimodal input format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    # Provide the raw uploaded image directly
                    "image": {
                        "type": "input_image",
                        "data": image_bytes
                    }
                },
                {
                    # Editing prompt provided by the user
                    "text": prompt
                }
            ]
        }
    ]

    # Call Qwen Image Edit model from DashScope
    response = MultiModalConversation.call(
        api_key=API_KEY,
        model="qwen-image-edit",
        messages=messages,
        stream=False,     # Non-streaming output
        n=1,              # Number of output images
        watermark=False,
        prompt_extend=True,
    )

    # If request succeeded
    if response.status_code == 200:
        # Extract the generated image URL from the response
        generated_image_url = response.output.choices[0].message.content[0]["image"]

        return {
            "success": True,
            "image_url": generated_image_url
        }

    # If failed, return error details
    return {
        "success": False,
        "status_code": response.status_code,
        "error_message": response.message,
    }


# ----------------------------------------------------------
# Health check endpoint for Render/Railway
# ----------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}