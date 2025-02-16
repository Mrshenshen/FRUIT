import torch
import transformers
from PIL import Image

# from .otter_image.modeling_otter import OtterForConditionalGeneration
from otter_ai import OtterForConditionalGeneration
import mimetypes
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image
import sys
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

sys.path.append("../../src")
# make sure you can properly access the otter folder
from otter_ai import OtterForConditionalGeneration

# Disable warnings
requests.packages.urllib3.disable_warnings()

# ------------------- Utility Functions -------------------


def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


# ------------------- Image and Video Handling Functions -------------------


def extract_frames(video_path, num_frames=16):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames // num_frames
    frames = []

    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).convert("RGB")
            frames.append(frame)

    video.release()
    return frames


def get_image(url: str) -> Union[Image.Image, list]:
    if "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    elif "video" in content_type:
        video_path = "temp_video.mp4"
        if "://" not in url:  # Local file
            video_path = url
        else:  # Remote URL
            with open(video_path, "wb") as f:
                f.write(requests.get(url, stream=True, verify=False).content)
        frames = extract_frames(video_path)
        if "://" in url:  # Only remove the temporary video file if it was downloaded
            os.remove(video_path)
        return frames
    else:
        raise ValueError("Invalid content type. Expected image or video.")


# ------------------- OTTER Prompt and Response Functions -------------------


def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"


def get_response(input_data, prompt: str, model=None, image_processor=None, tensor_dtype=None) -> str:
    if isinstance(input_data, Image.Image):
        vision_x = image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    elif isinstance(input_data, list):  # list of video frames
        vision_x = image_processor.preprocess(input_data, return_tensors="pt")["pixel_values"].unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image or list of video frames.")

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )

    # Get the data type from model's parameters
    model_dtype = next(model.parameters()).dtype

    # Convert tensors to the model's data type
    vision_x = vision_x.to(dtype=model_dtype)
    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]

    bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
        bad_words_ids=bad_words_id,
    )
    parsed_output = model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
    return parsed_output


# ------------------- Main Function -------------------
load_bit = "fp32"
if load_bit == "fp16":
    precision = {"torch_dtype": torch.float16}
elif load_bit == "bf16":
    precision = {"torch_dtype": torch.bfloat16}
elif load_bit == "fp32":
    precision = {"torch_dtype": torch.float32}

# This model version is trained on MIMIC-IT DC dataset.
model = OtterForConditionalGeneration.from_pretrained(r'./VLMs/otter/OTTER-Video-LLaMA7B-DenseCaption').to('cuda')
tensor_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[load_bit]

model.text_tokenizer.padding_side = "left"
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()
model.eval()




#EgoTV
log_file = open("/hzy_res/log_hzy_test_ego.txt", "w")

for task_type in['ns','nt','nsc','abs']:
    if task_type == 'ns':
        filename = "/egotv/ego/EgoTV/ns.json" # 700
    elif task_type == 'nt':
        filename = "/egotv/ego/EgoTV/nt.json" # 1080
    elif task_type == 'nsc':
        filename = "/egotv/ego/EgoTV/nsc.json" # 2164
    elif task_type == 'abs':
        filename = "/egotv/ego/EgoTV/abs.json" # 676

    with open(filename,'r') as file1:
        name_list = json.load(file1)
        
    with open("/egotv/csv/label_bank_aug.json",'r') as file2:
        hyps = json.load(file2)

        


    labels = [] 
    predictions = []
    count = 0
    log_file.write(task_type)
    log_file.write('\n')

    for item in tqdm(name_list):
        
        label = item[1]
        label = int(label)
        labels.append(label)
        hyp = item[2] + '. '

        prompt = 'Answer me yes or no: Do the following steps match the video actions? ' + hyp
        video_path= item[0] +"/video.mp4"


        frames_list = get_image(video_path)
        # try:
        response = get_response(frames_list, prompt, model, image_processor, tensor_dtype)
        log_file.write(video_path)
        log_file.write('\n')
        log_file.write(response)
        log_file.write('\n')
        log_file.flush()
        response = response.lower()
        if "yes" in response:
            pred = 1
        elif "no" or "not" in response:
            pred = 0
        else:
            raise ValueError()
        predictions.append(pred)
        # except:
        #     if len(labels) - 1 == len(predictions):
        #         labels.pop()
        #     count = count + 1




    assert len(labels)==len(predictions)
    acc = accuracy_score(labels, predictions)  # 计算准确率
    f1 = f1_score(labels, predictions)  # 计算F1分数
    print(task_type)
    print('ACC:%f   F1:%f'%(acc,f1))
    print('Untest num: %d' %count)
    log_file.write("ACC: %f,  F1:%f\n"%(acc,f1))
    log_file.write('Untest num: %d\n' %count)
    log_file.flush()



