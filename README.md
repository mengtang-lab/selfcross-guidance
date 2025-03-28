# Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects (CVPR2025)

> Weimin Qiu, Jieke Wang, Meng Tang  
> University of California Merced

[arXiv](https://arxiv.org/abs/2411.18936)

Diffusion models have achieved unprecedented fidelity and diversity for synthesizing image, video, 3D assets, etc. However, subject mixing is a known and unresolved issue for diffusion-based image synthesis, particularly for synthesizing multiple similar-looking subjects. We propose Self-Cross diffusion guidance to penalize the overlap between cross-attention maps and aggregated self-attention maps. Compared to previous methods based on self-attention or cross-attention alone, our self-cross guidance is more effective in eliminating subject mixing. What's more, our guidance addresses mixing for all relevant patches of a subject beyond the most discriminant one, e.g., beak of a bird. We aggregate self-attention maps of automatically selected patches for a subject to form a region that the whole subject attends to. Our method is training-free and can boost the performance of any transformer-based diffusion model such as Stable Diffusion. We also release a more challenging benchmark with many text prompts of similar-looking subjects and utilize GPT-4o for automatic and reliable evaluation. Extensive qualitative and quantitative results demonstrate the effectiveness of our Self-Cross guidance.


![Example images using different methods](examples.png "Example images using different methods")



## Description  
Official implementation of our Self-Cross Diffusion Guidance paper. 

![More example images using different methods](example2.png "More example images using different methods")

## Setup

### Dataset  
In the prompt.txt file, we have a new dataset for text-to-image synthesis. It contains two subsets with similar subjects. [SSD-2](https://github.com/mengtang-lab/selfcross-guidance/blob/main/prompts.txt) contains prompts with two similar subjects. [SSD-3](https://github.com/mengtang-lab/selfcross-guidance/blob/main/prompts.txt) contains prompts with three similar subjects.


### Environment
for generation
```
conda env create -f environment.yaml
conda activate selfcross
```
for eval
```
conda env create -f eval.yaml
conda activate lavis
```

### Hugging Face Diffusers Library
Our code relies also on Hugging Face's [diffusers](https://github.com/huggingface/diffusers) library for downloading the Stable Diffusion models. 


## Usage


To generate an image, you can simply run the `run_selfcross+initno.py` or `run_selfcross+conform.py` script. For example,
```
python run_selfcross+initno.py 
```

After generation, you can evaluate images with the `compute_text-to-image_similarity.py` or `compute_text-to-text_similarity.py` script. For example,
```
python compute_text-to-image_similarity.py
```

## Evaluate with GPT4o

To evaluate with GPT4o, enter the `API_KEY` in `eval_gpt4o.py` then run
```
python eval_gpt4o.py --dataset animal-animal --image_root DIR_TO_IMAGES  --output_dir OUTPUT_DIR 
```
Possible options for `--dataset` are `animal-animal`, `animal-object`, `object-object`, and `similar-subjects`. 
After the yaml file is generated, you can run statistics to get the numbers in Table 1
```
python eval_yaml.py --yaml_filename YAML_FILENAME
```
We also upload the yaml files under the directory eval_results.


Notes:

- To apply Stable Diffusion 2.1, specify: `model_choice = "SD21"` at the beginning of `run_selfcross+initno.py` or `run_selfcross+conform.py` script.
- You may want to change the seeds by rewriting the seeds.txt.
- You may want to change the prompts by rewriting the prompts.txt.
- Currently, the indices are set automatically according to the dataset provided by Attend&Excite paper. You may want to try different indices based on your own datasets. To do this, you can change the implementation of `token_groups` in `run_selfcross+conform.py` script, or `token_indices` in `run_selfcross+initno.py` in the main() function.
- if you would like to try stable diffusion only set `run_sd = True` in `run_selfcross+initno.py` 

All generated images will be saved to the path `"outputs/{dataset name}/{prompt}"`. All results of attention maps will be saved to the path `"attentions/{dataset name}/{prompt}"`.


## Acknowledgements 
This code is built on the codes from [diffusers](https://github.com/huggingface/diffusers) library, [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt/), [Attend&Excite](https://github.com/yuval-alaluf/Attend-and-Excite), [CONFORM](https://github.com/gemlab-vt/CONFORM) and [INITNO](https://github.com/xiefan-guo/initno).

## Citation
If you use this code for your research, please cite the following work: 
```
@inproceedings{2025selfcross,
  author    = {Weimin, Qiu and Jieke, Wang and Meng, Tang},
  title     = {Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects},
  booktitle   = {CVPR},
  year      = {2025},
}
```
