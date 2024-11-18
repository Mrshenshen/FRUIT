import os
import json
import argparse
import datetime

import torch
import numpy as np

from models import get_model
from sample_dataset import sample_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import disable_caching
disable_caching()

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="BLIP2")
    parser.add_argument("--types", type=str, default="Activity,Forecast,Localization/location,Localization/spatial,Object/affordance,Object/attribute,Object/existence,Planning/assistance,Planning/navigation,Reasoning/comparing,Reasoning/counting,Reasoning/situated")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    return args


def eval_sample_dataset(dataset, dataset_name, max_sample_num=50, seed=0):
    if max_sample_num == -1:
        return dataset
    return sample_dataset(dataset, dataset_name, max_sample_num, seed)

def load_dataset(args,type):
    annotation_path = "./data/"+type+"/annotations.json"
    caption_path = "./captions/"+type+"/image_captions/caption.jsonl"
    with open(annotation_path, 'r') as f:
        dataset = json.load(f)      
    for i, d in enumerate(dataset):
        image_filename = d['image_path'][0].split('/')[-1]
        dataset[i]['images'] = os.path.join(os.path.dirname(annotation_path), 'images', image_filename)
        img_id = os.path.splitext(image_filename)[0]
        data_dir="./clip_ego/"+type+"/"+img_id+"/"
        dataset[i]['images_with_annotation'] = os.path.join(os.path.dirname(data_dir), image_filename)
        dataset[i]['clip0'] = os.path.join(os.path.dirname(data_dir), "clip0.jpg")
        dataset[i]['clip1'] = os.path.join(os.path.dirname(data_dir), "clip1.jpg")
        dataset[i]['clip2'] = os.path.join(os.path.dirname(data_dir), "clip2.jpg")
    id=0
    with open(caption_path, 'r') as caption_file:
        for line in caption_file:
            json_line = json.loads(line)
            dataset[id]['captions']=json_line["caption"]
            id = id+1

    return dataset

def get_generation_args(dataset_name):
    if "Planning" in dataset_name or "planning" in dataset_name:
        return {
            'max_new_tokens': 300,
            'planning': True
        }
    else:
        return {
            'max_new_tokens': 30,
            'planning': False
        }


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model_name = args.model_name
    types = list(args.types.split(','))  
    print(types)

    print(f"Running inference: {model_name}")

    if 'blip2' in model_name.lower() or 'llava' in model_name.lower():
        batch_size = 1
    else:
        batch_size = args.batch_size
    model = get_model(model_name, device=torch.device('cuda'))

    for type in types:
        dataset_name = type        
        
        dataset = load_dataset(args,type)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})

        model_answers = []
        ref_answers = []
        question_files = []
        q_id = 0
        for batch in tqdm(dataloader, desc=f"Running inference: {model_name} on  {dataset_name}"):
            questions = batch['question']
            print(questions)
            images = [batch['images'][0]+','+batch['images_with_annotation'][0]]
            print("images:",images)
            captions = batch["captions"][0]

            if args.batch_size == 1:
                output = model.generate(images[0], questions[0], captions, **get_generation_args(dataset_name))
                outputs = [output]
            else:
                outputs = model.batch_generate(images, questions, **get_generation_args(dataset_name))
            print("***output:***",outputs)            
            for i, (question, answer, pred) in enumerate(zip(batch['question'], batch['answer'], outputs)):
                model_answers.append({
                    'question_id': q_id,
                    'model_id': model_name,
                    'choices':[{'index': 0, "turns": [pred]}]
                })
                ref_answers.append({
                    'question_id': q_id,
                    'model_id': 'ground_truth',
                    'choices':[{'index': 0, "turns": [answer]}]
                })
                question_files.append({
                    'question_id': q_id,
                    'turns': [question]
                })
                q_id += 1
        torch.cuda.empty_cache()
        # del model



        result_folder = "./results/"+type
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        model_answer_folder = os.path.join(result_folder, 'model_answer')
        if not os.path.exists(model_answer_folder):
            os.makedirs(model_answer_folder)
        with open(os.path.join(model_answer_folder, f"{model_name}.jsonl"), 'w') as f:
            for pred in model_answers:
                f.write(json.dumps(pred) + '\n')
        
        ref_answer_folder = os.path.join(result_folder, 'reference_answer')
        if not os.path.exists(ref_answer_folder):
            os.makedirs(ref_answer_folder)
        with open(os.path.join(ref_answer_folder, "ground_truth.jsonl"), 'w') as f:
            for ref in ref_answers:
                f.write(json.dumps(ref) + '\n')
        
        with open(os.path.join(result_folder,  "question.jsonl"), 'w') as f:
            for q in question_files:
                f.write(json.dumps(q) + '\n')

        

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)


