# Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects

> Weimin Qiu, Jieke Wang, Meng Tang  
> University of California Merced

>Diffusion models have achieved unprecedented fidelity and diversity for synthesizing image, video, 3D assets, etc. However, subject mixing is a known and unresolved issue for diffusion-based image synthesis, particularly for synthesizing multiple similar-looking subjects. We propose Self-Cross diffusion guidance to penalize the overlap between cross-attention maps and aggregated self-attention maps. Compared to previous methods based on self-attention or cross-attention alone, our self-cross guidance is more effective in eliminating subject mixing. What's more, our guidance addresses mixing for all relevant patches of a subject beyond the most discriminant one, e.g., beak of a bird. We aggregate self-attention maps of automatically selected patches for a subject to form a region that the whole subject attends to. Our method is training-free and can boost the performance of any transformer-based diffusion model such as Stable Diffusion.% for synthesizing similar subjects. We also release a more challenging benchmark with many text prompts of similar-looking subjects and utilize GPT-4o for automatic and reliable evaluation. Extensive qualitative and quantitative results demonstrate the effectiveness of our Self-Cross guidance.

[PDF](https://arxiv.org/abs/2411.18936)



## Description  
Official implementation of our Self-Cross Diffusion Guidance paper. 

## Setup

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
python run_selfcross+initno.py 
```

Notes:

- To apply Stable Diffusion 2.1, specify: `model_choice = "SD21"` at the beginning of `run_selfcross+initno.py` or `run_selfcross+conform.py` script.
- You may want to change the seeds by rewriting the seeds.txt.
- You may want to change the prompts by rewriting the prompts.txt.
- Currently, the indices are set automatically according to the dataset provided by Attend&Excite paper. You may want to try different indices based on your own datasets. To do this, you can change `token_groups` in `run_selfcross+conform.py` script, or `token_indices` in `run_selfcross+initno.py`.

All generated images will be saved to the path `"outputs/{dataset name}/{prompt}"`. All results of attention maps will be saved to the path `"attentions/{dataset name}/{prompt}"`.


## Acknowledgements 
This code is built on the codes from [diffusers](https://github.com/huggingface/diffusers) library, [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt/), [Attend&Excite](https://github.com/yuval-alaluf/Attend-and-Excite), [CONFORM](https://github.com/gemlab-vt/CONFORM) and [INITNO](https://github.com/xiefan-guo/initno).

## Citation
If you use this code for your research, please cite the following work: 
```
@misc{selfcross,
      title={Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects}, 
      author={Weimin Qiu and Jieke Wang and Meng Tang},
      year={2024},
      eprint={2411.18936},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
