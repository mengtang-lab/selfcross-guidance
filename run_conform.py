import sys
import os
import json
import torch
import random
import numpy as np
from diffusers import StableDiffusionPipeline
from pipelines.pipeline_conform import ConformPipeline
sys.path.append('.')

model_choice    = "SD14" #, "SD15", "SD21"
SD14_VERSION    = "CompVis/stable-diffusion-v1-4"
SD15_VERSION    = "runwayml/stable-diffusion-v1-5"
SD21_VERSION    = "stabilityai/stable-diffusion-2-1"

if model_choice == "SD14":
    sd_pipeline = StableDiffusionPipeline.from_pretrained(torch_dtype=torch.float16,pretrained_model_name_or_path=SD14_VERSION).to("cuda:0")
    attn_res = (16, 16)
    max_iter_to_alter = 25 # according to original setting in CONFORM
elif model_choice == "SD15":
    sd_pipeline = StableDiffusionPipeline.from_pretrained(torch_dtype=torch.float16,pretrained_model_name_or_path=SD15_VERSION).to("cuda:0")
    attn_res = (16, 16)
    max_iter_to_alter = 25 # according to original setting in CONFORM
elif model_choice == "SD21":
    sd_pipeline = StableDiffusionPipeline.from_pretrained(torch_dtype=torch.float16,pretrained_model_name_or_path=SD21_VERSION).to("cuda:0")
    attn_res = (24, 24)
    max_iter_to_alter = 25 # according to original setting in CONFORM
else:
    raise Exception("model_choice should be among SD14 , SD15, SD21.")

pipeline = ConformPipeline(
    vae=sd_pipeline.vae,
    text_encoder=sd_pipeline.text_encoder,
    tokenizer=sd_pipeline.tokenizer,
    unet=sd_pipeline.unet,
    scheduler=sd_pipeline.scheduler,
    safety_checker=sd_pipeline.safety_checker,
    feature_extractor=sd_pipeline.feature_extractor,
)


refinement_steps = 20 # Number of refinement steps
scale_factor = 20 # Scale factor for the optimization step
iterative_refinement_steps = [0, 10, 20] # Iterative refinement steps # according to original setting in CONFORM
smoothing_sigma = 0.5 # Sigma for the smoothing kernel
smoothing_kernel_size = 3 # Kernel size for the smoothing kernel
temperature = 0.5 # Temperature for the contrastive loss
loss_fn = "ntxent" # Loss function to use
dataset = "animals"

def Convert(string):
    li = list(string.split(" "))
    return li

with open('prompts.txt') as f:
    data = f.read()
print("Data type before reconstruction : ", type(data))
# reconstructing the data as a dictionary
prompts = json.loads(data)
print("seeds Data type after reconstruction : ", type(prompts))
print(prompts)
prompts=prompts[dataset]

with open('seeds.txt') as f:
    data = f.read()
print("seeds Data type before reconstruction : ", type(data))
# reconstructing the data as a dictionary
seeds = json.loads(data)
print("seeds Data type after reconstruction : ", type(seeds))
print(seeds)

token_groups=[]
for PROMPT in prompts:
    print(PROMPT)

    words = Convert(PROMPT)
    if len(words) == 5:
        token_groups = [[2], [5]]
    elif len(words) == 6:
        token_groups = [[2], [5, 6]]
    elif len(words) == 7:
        token_groups = [[2, 3], [6, 7]]
    else:
        print(words, len(words))
    print(PROMPT, token_groups)

    ids = pipeline.tokenizer(PROMPT, split_special_tokens=False).input_ids
    indices = {
        i: tok
        for tok, i in zip(pipeline.tokenizer.convert_ids_to_tokens(ids), range(len(ids)))
    }
    print('indices', indices)
    path = './outputs/{}/{}'.format(dataset, PROMPT)
    os.makedirs('{:s}'.format(path), exist_ok=True)
    result_root = './attentions/{}/{}'.format(dataset, PROMPT)
    os.makedirs('{:s}'.format(result_root), exist_ok=True)
    for SEED in seeds:
        SEED = int(SEED)
        print('Seed ({}) Processing the ({}) prompt'.format(SEED, PROMPT))
        generator = torch.Generator("cuda").manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED)

        images = pipeline(prompt=PROMPT, model_choice=model_choice, token_groups=token_groups, guidance_scale=7.5,generator=generator,
            num_inference_steps=50, max_iter_to_alter=max_iter_to_alter, attn_res=attn_res,scale_factor=scale_factor,iterative_refinement_steps=iterative_refinement_steps,
            do_smoothing=True, smoothing_sigma=smoothing_sigma,smoothing_kernel_size=smoothing_kernel_size,
            temperature=temperature, refinement_steps=refinement_steps, softmax_normalize=False,softmax_normalize_attention_maps=False,
            add_previous_attention_maps=True, previous_attention_map_anchor_step=None,loss_fn=loss_fn,result_root=result_root, seed=SEED)
        images[0].save(path + f'/{SEED}.png')

