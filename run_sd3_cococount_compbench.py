import torch
# from pipelines.SD3pipelineHeadsExplore import StableDiffusion3Pipeline
# from pipelines.SD3pipelineLayerExplore import StableDiffusion3Pipeline
from pipelines.SD3pipelineDiversityGuidance import StableDiffusion3Pipeline
import json
import os
import sys
import diffusers
sys.setrecursionlimit(5000)
run_sd = False
run_initno = False # comparing with previous version the difference is no average over pairs of objects
prompt_eng = False
model_choice =  "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(model_choice,torch_dtype=torch.bfloat16,  # precision matters for guidance!!!!
                                                token="your huggingface access token")

pipe = pipe.to("cuda:0")
pipe.transformer.enable_gradient_checkpointing()

with open('./data/CoCoCount.json') as f:
    data = f.read()
print("Data type before reconstruction : ", type(data))
# reconstructing the data as a dictionary
data = json.loads(data)
print("Data type after reconstruction : ", type(data))
# 'prompt', 'object', 'object_plural', 'object_id', 'scene', 'number', 'int_number', 'seed'
for i in range(len(data)):
    # if data[i]['object_plural'] != "ties" : continue # and data[i]['object_plural'] != "cups"
    if data[i]['int_number'] != 5 : continue
    # if data[i]['object_plural'] != "birds": continue
    if data[i]['object_plural'] != "cows": continue
    # if data[i]['object_plural'] != "sheep": continue
    # if data[i]['object_plural'] != "cats": continue
    # if data[i]['object_plural'] != "bears": continue
    # if data[i]['seed'] != 520748:continue

    SEED = data[i]['seed']
    PROMPT = data[i]['prompt']
    path1 = './SD3/outputs'
    path2 = f'./SD3/results/{PROMPT}_{SEED}'
    # image_url = path1 + f"/{PROMPT}-{SEED}.png"
    # if os.path.exists(image_url): continue
    if not os.path.exists(path1): os.makedirs(path1)
    if not os.path.exists(path2): os.makedirs(path2)

    # if os.path.exists(path1 + f"/{PROMPT}-{SEED}.png"): continue

    new_string = PROMPT.replace(",", " ")
    words = new_string.split(" ")  # tokenizer treats comma as an individual token

    if isinstance(data[i]['object_plural'], list): # for T2I-comp-bench
        keywords = data[i]['object_plural']  # "could be single word like "gloves" only!"
    else: # for coCoCount
        keywords = data[i]['object_plural'].split(" ")  # "could be multiple words like "baseball gloves" "
    # print("keywords : ", keywords, "words : ",words)

    if prompt_eng:
        new_list = []
        for word in words:
            new_list.append(word)
            if word == keywords[-1]:
                new_list.append('with different appearance')
        # print(new_list)
        words = new_list

        PROMPT = " ".join(words)
        # print(PROMPT)

    token_indices = [words.index(word) + 1 for word in keywords]
    print(token_indices, len(words))
    print('Seed ({}) Processing the ({}) prompt'.format(SEED, PROMPT))
    generator = torch.Generator("cuda").manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(SEED) 
    torch.cuda.manual_seed_all(SEED)
    images = pipe(prompt=PROMPT, token_indices=token_indices, guidance_scale=4.5, generator=generator,
                num_inference_steps=28, min_iter_to_alter=-1, max_iter_to_alter=14, attention_res=64,
                from_where=[4,5,6,7,8,9,10,11,12], result_root=path2, attns_per_layer=4,
                int_number=data[i]['int_number'], model_choice=model_choice,
                K=1, seed=SEED, run_sd=run_sd, run_initno=run_initno,scale_factor=60)
    # print(images[0])  # sort of attn, 4 for SD3 and 6 for SD35
    images[0].save(path1 + f"/{PROMPT}-{SEED}.png")


