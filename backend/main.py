import os
import base64
from typing import Optional

import dashscope
from dashscope import MultiModalConversation
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

# ======================
# 基本配置
# ======================

app = FastAPI(title="Qwen Image Edit (DashScope)")

# 允许前端跨域访问（本地开发简单粗暴允许全部）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 使用新加坡区域的 DashScope 接口
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

# 从环境变量读取 DashScope API Key
# 注意：这里用的是 DASHSCOPE_API_KEY（你之前用的那个）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


def get_mime_type(file: UploadFile) -> str:
    """根据上传的文件类型推断 MIME 类型，默认用 image/png。"""
    if file.content_type and file.content_type.startswith("image/"):
        return file.content_type
    # fallback
    return "image/png"


@app.post("/api/edit")
async def edit_image(
    prompt: str = Form(...),
    file: UploadFile = File(...),
    n: int = Form(1),  # 生成图片数量，默认 1 张
):
    """
    前端调这个接口：
    - 上传图片 + prompt
    - 调用 qwen-image-edit
    - 返回生成图片的 URL
    """

    if not DASHSCOPE_API_KEY:
        return {
            "success": False,
            "error": "后端未配置 DASHSCOPE_API_KEY 环境变量，请先 export/setx 后重启服务。",
        }

    # 1. 读取上传图片字节
    image_bytes = await file.read()

    # 2. 转成 base64 + data URL（DashScope multimodal 支持 data:image;base64,...）
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = get_mime_type(file)
    image_data_url = f"data:{mime};base64,{b64}"

    # 3. 构造 DashScope 消息格式
    # 参考你之前的写法，只是把 image 从 URL 换成 data URL
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_data_url},
                {
                    "text": prompt,
                },
            ],
        }
    ]

    try:
        # 4. 调用 DashScope qwen-image-edit
        response = MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model="qwen-image-edit",
            messages=messages,
            stream=False,
            n=n,
            watermark=False,
            # 你可以在这里传 negative_prompt
            # negative_prompt="no extra sunlight, no resolution change",
            prompt_extend=True,
        )

        if response.status_code == 200:
            # qwen-image-edit 的返回结构：
            # response.output.choices[0].message.content 是一个列表
            # 每个元素都是 {"image": "<url>"}
            images = [
                c["image"] for c in response.output.choices[0].message.content
                if "image" in c
            ]
            if not images:
                return {
                    "success": False,
                    "error": "调用成功但未返回图片 URL，请检查响应结构。",
                }

            # 默认只用第一张
            return {
                "success": True,
                "image_url": images[0],
                "all_image_urls": images,
            }

        # 非 200，返回错误信息
        return {
            "success": False,
            "status_code": response.status_code,
            "error_code": response.code,
            "error_message": response.message,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"调用 DashScope 失败: {e}",
        }


@app.get("/health")
async def health():
    return {"status": "ok"}