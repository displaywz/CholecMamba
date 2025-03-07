import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
import torch
from PIL import Image
import numpy as np
import re
from typing import List, Dict

# LLaVA相关依赖
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

# 禁用torch初始化
disable_torch_init()

# 初始化FastAPI
app = FastAPI()

# 全局模型变量
model_path = "../CholecMamba/checkpoints/cholecmamba-sft"
tokenizer = None
model = None
image_processor = None
context_len = None

# 启动时加载模型
@app.on_event("startup")
async def load_model():
    global tokenizer, model, image_processor, context_len
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    model.get_vision_tower().cuda()

# 请求模型
class OpenAIRequest(BaseModel):
    messages: List[Dict[str, str]]
    image: str  # base64编码的图片

# 响应模型
class OpenAIResponse(BaseModel):
    id: str = "chatcmpl-default"
    object: str = "chat.completion"
    created: int = 0
    model: str = "cholecmamba"
    choices: List[Dict]
    usage: Dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def process_image(image_b64: str):
    try:
        # 解码base64图片
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # 图像预处理
        image = image.resize((256, 256), resample=Image.BICUBIC)
        image = np.asarray(image)
        
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        else:
            image = image.transpose((2, 0, 1))
            
        if (image > 1).any():
            image = image / 255.0
            
        image_tensor = torch.as_tensor(image.copy()).to(dtype=torch.float16).contiguous()
        image_tensor = image_tensor.to(model.device, dtype=torch.float32).unsqueeze(dim=0)
        return image_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像处理失败: {str(e)}")

def generate_prompt(qs: str):
    conv_mode = "llama3"
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

@app.post("/v1/chat/completions")
async def chat_completion(request: OpenAIRequest):
    try:
        # 获取最后一条用户消息
        last_message = [msg for msg in request.messages if msg["role"] == "user"][-1]
        question = last_message["content"]
        
        # 处理图像
        image_tensor = process_image(request.image)
        
        # 生成prompt
        prompt = generate_prompt(question)
        
        # 生成输入IDs
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        
        # 生成参数
        generation_params = {
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "num_beams": 1,
            "max_new_tokens": 4096,
            "use_cache": True
        }
        
        # 执行推理
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                **generation_params
            )
        
        # 解码结果
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = outputs.replace("<|eot_id|>", "").strip()
        
        # 构造OpenAI兼容响应
        return OpenAIResponse(
            choices=[{
                "message": {
                    "role": "assistant",
                    "content": outputs
                },
                "index": 0,
                "finish_reason": "stop"
            }]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)