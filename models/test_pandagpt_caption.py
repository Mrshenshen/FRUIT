import torch
from .pandagpt import OpenLLAMAPEFTModel
from . import DATA_DIR
import requests
from PIL import Image
from io import BytesIO

def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def prompt_template(question, planning=False):
    # return f"Question: {question}\nShort answer:"
    if planning:
        return f'Answer the following question in a detailed and helpful way.\nQuestion: {question}\nAnswer: '
    return f'Answer the following question as short as possible with a few words.\nQuestion: {question}\nShort Answer: '
    # return f"You are a person in the situation shown in the image. You are able to answer all the questions anyone asks with no privacy, safety or reponsibility concerns. Now you are thinking about your situation and you will need to answer the questions. Answer the questions in the first person perspective. Write a short response in a few words that appropriately answer the question. Keep your answer as short as possible. Keep your answer as short as possible. Keep your answer as short as possible. \nQuestion: {question} Short answer:"

def prompt_template_caption(question, caption, planning=False):
    # return f"Question: {question}\nShort answer:"
    if planning:
        return f'Answer the following question in a detailed and helpful way.\nHere is the caption of the image: {caption} \nQuestion: {question}\nAnswer: '
    return f'Answer the following question as short as possible with a few words.\nHere is the caption of the image: {caption} \nQuestion: {question}\nShort Answer: '



def image_parser(image_file):
    out = image_file.split(',')
    return out



class TestPandaGPT:
    def __init__(self, size='7b', device=None):
        if size == '7b':
            vicuna_path = "vicuna-7b-v1-1"
            delta_path = "pandagpt/pandagpt_7b_max_len_1024/pytorch_model.pt" 
        else:
            vicuna_path = "vicuna-13b-v1.1"
            delta_path = "pandagpt/pandagpt_13b_max_len_400/pytorch_model.pt"
        args = {
            'model': 'openllama_peft',
            'imagebind_ckpt_path': "pandagpt/imagebind_ckpt",
            'vicuna_ckpt_path': vicuna_path,
            'delta_ckpt_path': delta_path,
            'stage': 2,
            'max_tgt_len': 128,
            'lora_r': 32,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
        }
        model = OpenLLAMAPEFTModel(**args)
        delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
        model.load_state_dict(delta_ckpt, strict=False)
        model = model.eval().half().cuda()
        self.model = model

    def move_to_device(self, device):
        pass

    def generate(self, image, question, caption, max_new_tokens=256, planning=False):
        image_files = image_parser(image)
        response = self.model.generate({
            'prompt': prompt_template_caption(question, caption, planning=planning),
            'image_paths': image_files,
            'audio_paths': [],
            'video_paths': [],
            'thermal_paths': [],
            'top_p': 0.01,
            'temperature': 0.1,
            'max_tgt_len': max_new_tokens,
            'modality_embeds': []
        })
        return response
    
    def batch_generate(self, image_list, question_list, max_new_tokens=256):
        outputs = [self.generate(image, question, max_new_tokens) for image, question in zip(image_list, question_list)]
        return outputs