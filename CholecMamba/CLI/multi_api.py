import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from PIL import Image
import re
import numpy as np
import io
import base64
from typing import List
import json
from io import BytesIO

# Disable PyTorch initialization
disable_torch_init()

# Load LLaVA model
model_path = "../CholecMamba/checkpoints/cholecmamba-sft"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
model.get_vision_tower().cuda()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_single_image(image_data: str) -> torch.Tensor:
    image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
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

def process_multiple_images(image_data_list: List[str]) -> torch.Tensor:
    image_tensors = []
    for image_data in image_data_list:
        image_tensor = process_single_image(image_data)
        image_tensors.append(image_tensor.unsqueeze(dim=0))
        x = torch.cat(image_tensors, dim=1)
    print(x.shape)
    return x

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    messages = request["messages"]
    if not messages or messages[-1]["role"] != "user":
        return {"error": "Invalid messages format"}

    last_message = messages[-1]
    content = last_message["content"]
    
    # Extract text and images from the message
    if isinstance(content, list):
        text = next((item["text"] for item in content if item["type"] == "text"), "")
        image_data_list = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
    else:
        text = content
        image_data_list = []

    # Process images if present
    if image_data_list:
        image_tensors = process_multiple_images(image_data_list)
        num_images = len(image_data_list)
        
        # # Format prompt with image tokens for multiple images
        # if model.config.mm_use_im_start_end:
        #     image_tokens = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN) 
        # else:
        #     image_tokens = DEFAULT_IMAGE_TOKEN 
            
        # # Check if text contains IMAGE_PLACEHOLDER
        # if IMAGE_PLACEHOLDER in text:
        #     prompt = text.replace(IMAGE_PLACEHOLDER, image_tokens)
        # else:
        #     prompt = image_tokens + "\n" + text
        prompt = "<image>\n" + text    
        # Prepare conversation
        conv = conv_templates["llama3"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)
        # Tokenize input
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # Generate response
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                do_sample=True,
                temperature=0.9,
                max_new_tokens=4096,
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = outputs.replace("<|eot_id|>", "").strip()
        # Decode response
        response = outputs
    else:
        # Handle text-only input
        conv = conv_templates["llama3"].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                max_new_tokens=4096,
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = outputs.replace("<|eot_id|>", "").strip()
        # Decode response
        response = outputs

    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response
            }
        }]
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)