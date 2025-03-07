import streamlit as st
import time
from PIL import Image
import random
import tkinter as tk
from tkinter import filedialog
import os
import requests
from io import BytesIO
import re
import cv2
import numpy as np
import tempfile

# FastAPI service URLs
FASTAPI_URL = "http://localhost:8000/generate"
FASTAPI_MULTI_URL = "http://localhost:8000/generate_multi"

# Initialize session state
if "model_option" not in st.session_state:
    st.session_state.model_option = "CholecMamba"

def set_page_title():
    st.set_page_config(page_title="CholecMamba")
    st.title('CholecMamba')

set_page_title()

def extract_frames(video_file, num_frames=10):
    """Extract evenly spaced frames from a video file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(video_file.read())
            tmpfile.flush()
            
            cap = cv2.VideoCapture(tmpfile.name)
            
            if not cap.isOpened():
                raise ValueError("Error: Could not open video file")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError("Error: Could not get frame count")
            
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                else:
                    print(f"Warning: Could not read frame {idx}")
            
            cap.release()
            os.unlink(tmpfile.name)
            video_file.seek(0)
            
            if not frames:
                raise ValueError("Error: No frames could be extracted")
            
            return frames
            
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None

def process_message(prompt, media=None, is_video=False, chat_history=None):
    """Process message with segmentation support and chat history"""
    try:
        files = {}
        
        # 构建包含对话历史的完整提示
        if chat_history and len(chat_history) > 0:
            # 格式化对话历史
            formatted_history = ""
            for msg in chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                formatted_history += f"{role}: {msg.get('content', '')}\n\n"
            
            # 添加当前问题
            complete_prompt = f"{formatted_history}User: {prompt}\nAssistant:"
        else:
            complete_prompt = prompt
            
        data = {
            'prompt': complete_prompt,
            'seg_enabled': st.session_state.seg_enabled
        }
        
        if media is not None:
            if is_video:
                for i, frame in enumerate(media):
                    img_byte_arr = BytesIO()
                    frame.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    files[f'images'] = (f'frame_{i}.png', img_byte_arr, 'image/png')
                url = FASTAPI_MULTI_URL
            else:
                img_byte_arr = BytesIO()
                media.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                files = {'image': ('image.png', img_byte_arr, 'image/png')}
                url = FASTAPI_URL
        else:
            url = FASTAPI_URL
        
        response = requests.post(url, files=files, data=data, stream=True)
        
        if response.status_code == 200:
            normal_placeholder = st.empty()
            thought_placeholder = st.empty()
            solution_placeholder = st.empty()
            seg_placeholder = st.empty()  # Placeholder for segmentation results
            
            final_normal = ""
            final_thought = ""
            final_solution = ""
            buffer = ""
            state = "normal"
            
            begin_thought_pattern = re.compile(r"'?<\|begin_of_thought\|>'?")
            end_thought_pattern = re.compile(r"'?<\|end_of_thought\|>'?")
            begin_solution_pattern = re.compile(r"'?<\|begin_of_solution\|>'?")
            end_solution_pattern = re.compile(r"'?<\|end_of_solution\|>'?")

            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    # Check for segmentation results
                    if "Segmentation result saved as" in chunk and st.session_state.seg_enabled:
                        try:
                            # Wait briefly for the file to be saved
                            time.sleep(0.1)
                            seg_img = Image.open("seg_result.png")
                            seg_placeholder.image(seg_img, caption='Segmentation Result', use_column_width=True)
                            continue
                        except Exception as e:
                            seg_placeholder.error(f"Error loading segmentation: {str(e)}")
                            continue
                            
                    # Check for GIF segmentation results
                    if "Segmentation results saved as animated GIF:" in chunk and st.session_state.seg_enabled:
                        try:
                            # Extract GIF filename from chunk
                            gif_match = re.search(r"animated GIF: ([^\s]+)", chunk)
                            if gif_match:
                                gif_filename = gif_match.group(1)
                                # Wait briefly for the file to be saved
                                time.sleep(0.5)
                                gif_img = Image.open(gif_filename)
                                seg_placeholder.image(gif_img, caption='Segmentation Result (GIF)', use_column_width=True)
                        except Exception as e:
                            seg_placeholder.error(f"Error loading GIF segmentation: {str(e)}")
                            continue

                    buffer += chunk
                    while True:
                        if state == "normal":
                            thought_match = begin_thought_pattern.search(buffer)
                            solution_match = begin_solution_pattern.search(buffer)

                            next_match = None
                            if thought_match and (not solution_match or thought_match.start() < solution_match.start()):
                                next_match = ("thought", thought_match)
                            elif solution_match:
                                next_match = ("solution", solution_match)

                            if next_match:
                                section, match = next_match
                                final_normal += buffer[:match.start()]
                                normal_placeholder.markdown(final_normal)
                                buffer = buffer[match.end():]
                                state = section
                            else:
                                final_normal += buffer
                                normal_placeholder.markdown(final_normal)
                                buffer = ""
                                break

                        elif state == "thought":
                            end_match = end_thought_pattern.search(buffer)
                            if end_match:
                                final_thought += buffer[:end_match.start()]
                                thought_placeholder.info(f'思考："{final_thought}"')
                                buffer = buffer[end_match.end():]
                                state = "normal"
                            else:
                                final_thought += buffer
                                thought_placeholder.info(f'思考："{final_thought}"')
                                buffer = ""
                                break

                        elif state == "solution":
                            end_match = end_solution_pattern.search(buffer)
                            if end_match:
                                final_solution += buffer[:end_match.start()]
                                solution_placeholder.markdown(final_solution)
                                buffer = buffer[end_match.end():]
                                state = "normal"
                            else:
                                final_solution += buffer
                                solution_placeholder.markdown(final_solution)
                                buffer = ""
                                break

            final_normal = final_normal.replace("'", "")
            final_thought = final_thought.replace("'", "")
            final_solution = final_solution.replace("'", "")
            
            if buffer:
                if state == "normal":
                    final_normal += buffer
                    normal_placeholder.markdown(final_normal)
                elif state == "thought":
                    final_thought += buffer
                    thought_placeholder.info(f'思考："{final_thought}"')
                elif state == "solution":
                    final_solution += buffer
                    solution_placeholder.markdown(final_solution)
            
            return {"content": final_normal, "thought": final_thought, "solution": final_solution}
        else:
            return {"content": f"Error: {response.status_code} - {response.text}", "thought": "", "solution": ""}
    except Exception as e:
        return {"content": f"Error connecting to FastAPI service: {str(e)}", "thought": "", "solution": ""}

def render_file_uploaders():
    """Handle file uploads for both images and videos"""
    uploaded_media = None
    is_video = False
    
    upload_type = st.radio("Upload type", ["Image", "Video"])
    
    if upload_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg', 'bmp'])
        if uploaded_file is not None:
            try:
                uploaded_media = Image.open(uploaded_file)
                st.image(uploaded_media, caption='Input image', use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    elif upload_type == "Video":
        uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            frames = extract_frames(uploaded_file)
            if frames:
                uploaded_media = frames
                is_video = True
                
                with BytesIO() as gif_buffer:
                    first_frame = frames[0].convert('RGB')
                    frames_rgb = [frame.convert('RGB').resize(first_frame.size) for frame in frames]
                    
                    frames_rgb[0].save(
                        gif_buffer,
                        format='GIF',
                        save_all=True,
                        append_images=frames_rgb[1:],
                        duration=500,
                        loop=0,
                        optimize=True
                    )
                    gif_buffer.seek(0)
                    
                    st.image(
                        gif_buffer,
                        caption='Extracted Frames Animation',
                        use_column_width=True,
                        output_format='GIF'
                    )
    
    return uploaded_media, is_video

# Sidebar configuration
with st.sidebar:
    st.subheader('Parameters')
    selected_model = 'CholecMamba'
    
    if 'seg_enabled' not in st.session_state:
        st.session_state.seg_enabled = False
    st.session_state.seg_enabled = st.checkbox('Seg', value=st.session_state.seg_enabled)
    
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "I am CholecMamba 🤖, here to provide assistance and advice for gallbladder surgery, along with video and image analysis understanding! 💡🔍"}]
    
    st.sidebar.button('Clear history', on_click=clear_chat_history)
    temperature = st.slider('temperature', min_value=0.01, max_value=1.0, value=0.8, step=0.01)
    top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('max_length', min_value=32, max_value=4096, value=2048, step=8)
    st.markdown('🔗 Learn about the creation of CholecMamba [team](http://www.radiomics.net.cn/change-language/en)💡')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am CholecMamba 🤖, here to provide assistance and advice for gallbladder surgery, along with video and image analysis understanding! 💡🔍"}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and ("thought" in message and message["thought"]):
            st.info(f'思考："{message["thought"]}"')
            st.write(message.get("solution", ""))
            st.markdown(message.get("content", ""))
        else:
            st.write(message.get("content", ""))

# Handle file uploads
uploaded_media, is_video = render_file_uploaders()

# Chat input and processing
if prompt := st.chat_input("Enter your message here..."):
    # 添加seg_system前缀（如果启用）
    processed_prompt = f"<seg_system>{prompt}" if st.session_state.seg_enabled else prompt
    
    # 保存用户消息到会话状态
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # 获取最近的对话历史（最多取最后10条消息以避免上下文过长）
    chat_history = st.session_state.messages[-11:-1] if len(st.session_state.messages) > 11 else st.session_state.messages[:-1]
    
    # 处理响应，传递对话历史
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response_dict = process_message(processed_prompt, uploaded_media, is_video, chat_history)
            message = {
                "role": "assistant",
                "content": response_dict["content"],
                "thought": response_dict["thought"],
                "solution": response_dict["solution"]
            }
            st.session_state.messages.append(message)