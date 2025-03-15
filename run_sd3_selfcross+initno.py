import torch
from piplines.sd3pipline import StableDiffusion3Pipeline
import json
import os
import sys
sys.setrecursionlimit(5000)
run_sd  =  False
run_initno = True
model_choice = "stabilityai/stable-diffusion-3-medium-diffusers"  #"stabilityai/stable-diffusion-3.5-medium", #
dataset= 'TSD'
def Convert(string):
    li = list(string.split(" "))
    return li

pipe = StableDiffusion3Pipeline.from_pretrained(model_choice,
    torch_dtype=torch.bfloat16,
    token = "your huggingface user token"
)
pipe = pipe.to("cuda:0")

with open('prompt.txt') as f:
    data = f.read()
print("Data type before reconstruction : ", type(data))
# reconstructing the data as a dictionary
prompts = json.loads(data)
print("seeds Data type after reconstruction : ", type(prompts))
print(prompts)
prompts = prompts[dataset]

with open('seeds.txt') as f:
    data = f.read()
print("seeds Data type before reconstruction : ", type(data))
# reconstructing the data as a dictionary
seeds = json.loads(data)
print("seeds Data type after reconstruction : ", type(seeds))
print(seeds)

for PROMPT in prompts:
    path1 = './SD3/outputs/{}'.format(PROMPT)
    path2 = './SD3/results/{}'.format(PROMPT)
    words = Convert(PROMPT)
    token_indices = [2, 5, len(words)]
    print(PROMPT, token_indices)
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    for SEED in seeds:
        SEED = int(SEED)
        print('Seed ({}) Processing the ({}) prompt'.format(SEED, PROMPT))
        generator = torch.Generator("cuda").manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        images = pipe(prompt=PROMPT, token_indices=token_indices, guidance_scale=4.5, generator=generator,
                      num_inference_steps=28, max_iter_to_alter=14,attention_res = 64, from_where=[5,6,7,8,9,10,11,12],
                      result_root=path2, K=16, seed=SEED, run_sd=run_sd, run_initno=run_initno).images
        images[0].save(path1 + f"/{SEED}.png")


