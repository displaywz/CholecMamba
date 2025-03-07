import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from PIL import Image
import re
import numpy as np
from typing import Optional, List
import io
from threading import Thread
from transformers.generation.streamers import TextIteratorStreamer
import matplotlib.pyplot as plt
from swin_umamba import SwinUMamba

# Disable PyTorch initialization
disable_torch_init()

# Load LLaVA model
model_path = "../CholecMamba/checkpoints/cholecmamba-sft"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
model.get_vision_tower().cuda()

# Initialize FastAPI
app = FastAPI(
    title="CholecMamba API",
    description="API for CholecMamba model",
    version="1.0.0",
)

# Initialize segmentation model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_model = SwinUMamba(
    in_chans=3,
    out_chans=13,
    feat_size=[64, 128, 256, 512, 1024],
    deep_supervision=False,
    hidden_size=1024,
)
seg_pth_path = "../Model/SwinUMamba.pth"
seg_state_dict = torch.load(seg_pth_path, map_location='cpu')
seg_model.load_state_dict(seg_state_dict, strict=False)
seg_model.eval()
seg_model = seg_model.to(device)

# Segmentation class mapping
class_mapping = {
    "Liver Ligament": 0,
    "Abdominal Wall": 1,
    "Fat": 2,
    "Gastrointestinal Tract": 3,
    "Liver": 4,
    "Gallbladder": 5,
    "Connective Tissue": 6,
    "Blood": 7,
    "Cystic Duct": 8,
    "Grasper": 9,
    "L-hook Electrocautery": 10,
    "Hepatic Vein": 11,
    "Black Background": 12,
}

def process_image(image_file: bytes) -> (torch.Tensor, tuple):
    """Process a single image and return tensor along with original dimensions"""
    image = Image.open(io.BytesIO(image_file)).convert('RGB')
    original_size = image.size  # Save original dimensions (width, height)
    
    # Resize for model processing
    resized_image = image.resize((256, 256), resample=Image.BICUBIC)
    image_array = np.asarray(resized_image)
    
    if image_array.ndim == 2:
        image_array = image_array[np.newaxis, ...]
    else:
        image_array = image_array.transpose((2, 0, 1))
    
    if (image_array > 1).any():
        image_array = image_array / 255.0
    
    image_tensor = torch.as_tensor(image_array.copy()).to(dtype=torch.float16).contiguous()
    image_tensor = image_tensor.to(model.device, dtype=torch.float32).unsqueeze(dim=0)
    
    return image_tensor, original_size

def process_multiple_images(image_files: List[bytes]) -> (torch.Tensor, List[tuple]):
    """Process multiple images and return combined tensor with original dimensions"""
    images = []
    original_sizes = []
    
    for image_file in image_files:
        image = Image.open(io.BytesIO(image_file)).convert('RGB')
        original_sizes.append(image.size)  # Save original dimensions
        
        resized_image = image.resize((256, 256), resample=Image.BICUBIC)
        image_array = np.asarray(resized_image)
        
        if image_array.ndim == 2:
            image_array = image_array[np.newaxis, ...]
        else:
            image_array = image_array.transpose((2, 0, 1))
        
        if (image_array > 1).any():
            image_array = image_array / 255.0
            
        image_tensor = torch.as_tensor(image_array.copy()).to(dtype=torch.float16).contiguous()
        image_tensor = image_tensor.to(model.device, dtype=torch.float32).unsqueeze(dim=0)
        image_tensor = image_tensor.unsqueeze(dim=1)
        images.append(image_tensor)
    
    return torch.cat(images, dim=1), original_sizes

def extract_seg_class(text: str) -> Optional[str]:
    """Extract segmentation class from model response"""
    match = re.search(r'→.*?<(.*?)>', text)
    if match:
        class_name = match.group(1)
        # Map class names to exact matches in class_mapping
        name_mapping = {
            "Liver": "Liver",
            "GallBladder": "Gallbladder",
            "Gall Bladder": "Gallbladder",
            # Add more mappings as needed
        }
        return name_mapping.get(class_name, class_name)
    match = re.search(r'->.*?<(.*?)>', text)
    if match:
        class_name = match.group(1)
        # Map class names to exact matches in class_mapping
        name_mapping = {
            "Liver": "Liver",
            "GallBladder": "Gallbladder",
            "Gall Bladder": "Gallbladder",
            # Add more mappings as needed
        }
        return name_mapping.get(class_name, class_name)
    return None

def perform_segmentation(image_tensor: torch.Tensor, class_name: str, original_size: tuple) -> Optional[str]:
    """Perform segmentation on the image and resize back to original dimensions"""
    try:
        # Get segmentation class index
        class_idx = class_mapping.get(class_name)
        if class_idx is None:
            return None

        # Ensure image is correctly formatted
        if image_tensor.dim() == 4:  # (B, C, H, W)
            input_tensor = image_tensor
        else:
            input_tensor = image_tensor.unsqueeze(0)

        # Perform segmentation at 256x256 resolution
        with torch.no_grad():
            output = seg_model(input_tensor)
        
        # Get prediction mask
        pred_mask = output.argmax(dim=1)
        
        # Create highlighted mask
        highlighted_mask = (pred_mask == class_idx)
        
        # Convert to numpy for visualization
        original_image = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        mask = highlighted_mask.squeeze().cpu().numpy()
        
        # Create semi-transparent overlay instead of solid color
        # Create a blend of original image and yellow highlight
        overlay = original_image.copy()
        yellow_highlight = np.zeros_like(original_image)
        yellow_highlight[mask] = [1, 1, 0]  # Yellow highlight
        
        # Apply semi-transparency (alpha=0.5)
        alpha = 0.5
        overlay = original_image * (1 - alpha * mask[:, :, np.newaxis]) + yellow_highlight * alpha * mask[:, :, np.newaxis]
        
        # Resize back to original dimensions before saving
        overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
        overlay_pil = overlay_pil.resize(original_size, resample=Image.BICUBIC)
        
        # Save result
        plt.imsave("seg_result.png", np.array(overlay_pil)/255.0)
        return "seg_result.png"
        
    except Exception as e:
        print(f"Segmentation error: {str(e)}")
        return None

def stream_generate(prompt: str, image_data: Optional[tuple], seg_enabled: bool = False):
    """Stream generation with segmentation support"""
    conv_mode = "llama3"
    accumulated_text = ""
    
    if image_data is not None:
        image_tensor, original_size = image_data
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in prompt:
            if model.config.mm_use_im_start_end:
                prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt)
            else:
                prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
        else:
            if model.config.mm_use_im_start_end:
                prompt = image_token_se + "\n" + prompt
            else:
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
                
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(final_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    else:
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()
        input_ids = tokenizer(text=final_prompt, return_tensors="pt").to(model.device)['input_ids']
    
    temperature = 0.8
    top_p = 0.9
    max_new_tokens = 4096

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)
    thread = Thread(target=model.generate, kwargs=dict(
        inputs=input_ids,
        images=image_data[0] if image_data is not None else None,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        use_cache=True,
    ))
    thread.start()
    
    for new_text in streamer:
        accumulated_text += new_text
        yield new_text
        
        # Check for segmentation command if enabled
        if seg_enabled and image_data is not None:
            seg_class = extract_seg_class(accumulated_text)
            if seg_class:
                seg_result = perform_segmentation(image_data[0], seg_class, image_data[1])
                if seg_result:
                    yield f"\nSegmentation result saved as {seg_result}"


def stream_generate_multi(prompt: str, image_data: Optional[tuple], seg_enabled: bool = False):
    """Stream generation for multiple images with segmentation support"""
    conv_mode = "llama3"
    accumulated_text = ""
    
    if image_data is not None:
        images_tensor, original_sizes = image_data
        num_images = images_tensor.size(1)
        image_tokens = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN) * num_images
        
        if IMAGE_PLACEHOLDER in prompt:
            if model.config.mm_use_im_start_end:
                prompt = re.sub(IMAGE_PLACEHOLDER, image_tokens, prompt)
            else:
                prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN * num_images, prompt)
        else:
            if model.config.mm_use_im_start_end:
                prompt = image_tokens + "\n" + prompt
            else:
                prompt = (DEFAULT_IMAGE_TOKEN * num_images) + "\n" + prompt
                
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(final_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    else:
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()
        input_ids = tokenizer(final_prompt, return_tensors="pt").to(model.device)['input_ids']
    
    temperature = 0.8
    top_p = 0.9
    max_new_tokens = 4096

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)
    thread = Thread(target=model.generate, kwargs=dict(
        inputs=input_ids,
        images=image_data[0] if image_data is not None else None,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        use_cache=True,
    ))
    thread.start()
    
    for new_text in streamer:
        accumulated_text += new_text
        yield new_text
        
        # Handle segmentation for each image if enabled
        if seg_enabled and image_data is not None:
            seg_class = extract_seg_class(accumulated_text)
            if seg_class:
                for i in range(image_data[0].size(1)):
                    single_image = image_data[0][:, i:i+1].squeeze(1)
                    original_size = image_data[1][i]
                    seg_result = perform_segmentation(single_image, seg_class, original_size)
                    if seg_result:
                        yield f"\nSegmentation result for image {i+1} saved as {seg_result}"

@app.post("/generate")
async def generate(
    image: Optional[UploadFile] = File(None),
    prompt: str = Form(...),
    seg_enabled: bool = Form(False)
):
    """Handle single image generation with segmentation support"""
    print(f"Received prompt: {prompt}")
    print(f"Segmentation enabled: {seg_enabled}")
    
    try:
        if image:
            print(f"Received image filename: {image.filename}")
            print(f"Received image content_type: {image.content_type}")
            image_file = await image.read()
            print(f"Successfully read image file, size: {len(image_file)} bytes")
            image_tensor, original_size = process_image(image_file)
            print(f"Image tensor shape: {image_tensor.shape}, Original size: {original_size}")
            return StreamingResponse(
                stream_generate(prompt, (image_tensor, original_size), seg_enabled),
                media_type="text/plain"
            )
        else:
            print("No image provided, processing text-only prompt")
            return StreamingResponse(
                stream_generate(prompt, None, seg_enabled),
                media_type="text/plain"
            )
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_multi")
async def generate_multi(
    prompt: str = Form(...),
    images: List[UploadFile] = File(None),
    seg_enabled: bool = Form(False)
):
    """Handle multiple image generation with segmentation support"""
    print(f"Received prompt for multiple images: {prompt}")
    print(f"Segmentation enabled: {seg_enabled}")
    
    try:
        if images and len(images) > 0:
            print(f"Received {len(images)} images")
            image_files = []
            for img in images:
                print(f"Processing image: {img.filename}")
                image_files.append(await img.read())
            
            image_tensor, original_sizes = process_multiple_images(image_files)
            print(f"Combined image tensor shape: {image_tensor.shape}")
            print(f"Original sizes: {original_sizes}")
            return StreamingResponse(
                stream_generate_multi(prompt, (image_tensor, original_sizes), seg_enabled),
                media_type="text/plain"
            )
        else:
            print("No images provided, processing text-only prompt")
            return StreamingResponse(
                stream_generate_multi(prompt, None, seg_enabled),
                media_type="text/plain"
            )
    except Exception as e:
        print(f"Error occurred in multi-image processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)